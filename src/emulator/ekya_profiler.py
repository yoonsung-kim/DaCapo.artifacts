import copy
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from scipy.optimize import curve_fit
from typing import List, Tuple, Dict, Any
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from bfp.bfp_model_precision_changer import BfpModelPrecisionChanger
from util.accuracy import calculate_accuracy, ClassificationAccuracyTracker
from emulator.profiler import Profiler, ProfileResult, ModelPrecision, TRAIN, INFERENCE, LABEL


MAX_EPOCH = 10

EPOCH_CANDIDATES = (np.arange(9) + 2).tolist()

PROFILE_SCALE = 0.1

LABEL_SAMPLING_RATES = {
    "resnet18": 0.20750,
    "resnet34": 0.09611,
    "vit_b_32": 0.2
}

TRAIN_SAMPLING_RATE_CANDIDATES = {
    "resnet18": [
        0.23025,
        0.61580,
        1.
    ],
    "resnet34": [
        0.23988,
        0.63873,
        1.
    ],
    "vit_b_32": [
        0.2,
        0.5,
        1.
    ]
}


def split_indices(indices: List[int], num_to_split: int) -> List[List[int]]:
    num_indices = len(indices)
    num_remained = num_indices - num_to_split

    assert num_indices > num_to_split
    assert num_to_split >= num_remained

    scale = int(math.floor(num_to_split / num_remained))

    idx = 0
    cnt_remained = 0

    splitted_indices = []
    remained_indices = []

    while cnt_remained != num_remained:
        remained_indices.append(indices[idx])
        idx += 1
        cnt_remained += 1

        splitted_indices.extend(indices[idx:idx+scale])
        idx += scale

    splitted_indices.extend(indices[idx:])

    return splitted_indices, remained_indices


def scipy_fit(func, xp, yp, sigma=None):
    popt, _ = curve_fit(func, xp, yp, sigma=sigma, method='dogbox', absolute_sigma=True)
    return lambda x: func(x, *popt)


def curve(x, b0, b1, b2):
    return 1 - (1 / (b0 * x + b1) + b2)


def get_expected_accuracy(before_train_acc, after_train_acc, microprofile_epoch, epoch):
    END_EPOCH = MAX_EPOCH
    END_TARGET_ACC = 0.95
    MICROPROFILE_EXPECTATION_FACTOR = 0.95

    seed_x = np.array([0, END_EPOCH])
    seed_y = np.array([before_train_acc, END_TARGET_ACC])
    seed_curve = scipy_fit(curve, seed_x, seed_y)
    
    microprofile_expected_values = seed_curve(microprofile_epoch)
    microprofile_deviation = min(after_train_acc / (microprofile_expected_values * MICROPROFILE_EXPECTATION_FACTOR), 1)
    new_end_acc = END_TARGET_ACC * microprofile_deviation
    
    new_seed_y = np.array([before_train_acc, new_end_acc])
    seed_curve = scipy_fit(curve, seed_x, new_seed_y)

    booster_pts_x = np.linspace(min(seed_x), max(seed_x), 4)
    booster_pts_y = seed_curve(booster_pts_x)

    xp = np.concatenate([booster_pts_x, seed_x, [microprofile_epoch,]])
    yp = np.concatenate([booster_pts_y, new_seed_y, [after_train_acc,]])
    
    try:
        fn = scipy_fit(curve, xp, yp)
        return fn(epoch)
    except:
        return after_train_acc


class CustomSampler(Sampler):
    def __init__(self, indices, shuffle: bool = False):
        self.indices = indices
        self.shuffle = shuffle

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class EkyaProfiler(Profiler):
    def __init__(self,
                 config):
        super().__init__(config)

        self.is_first = True

        # self.static_epochs = self.config.initial_epoch
        self.static_m_T_p1 = ModelPrecision.MX9
        self.static_m_I_p1 = self.config.initial_m_I_p1
        self.static_m_L_p2 = ModelPrecision.MX6
        self.static_m_I_p2 = self.config.initial_m_I_p2
        self.static_f_I_p1 = self.config.initial_f_I_p1
        self.static_f_I_p2 = self.config.initial_f_I_p2
        # self.static_sampling_rate = self.config.initial_sampling_rate
        self.micro_profile_epoch = 5
        self.precision_changer = BfpModelPrecisionChanger()

        self.label_srs = copy.deepcopy(LABEL_SAMPLING_RATES)
        self.train_sr_candidates = copy.deepcopy(TRAIN_SAMPLING_RATE_CANDIDATES)

        # self.label_srs["vit_b_32"] = vit_sr
        # self.train_sr_candidates["vit_b_32"][0] = vit_t_sr_m
        # self.train_sr_candidates["vit_b_32"][1] = vit_t_sr_s

    def __micro_profile(self,
                        module: nn.Module,
                        entire_dataset: Dataset,
                        indices_to_micro_profile: List[int]) -> List[Dict[str, Any]]:
        device = self.config.profile_device
        num_indices = len(indices_to_micro_profile)
        sr_candidates = self.train_sr_candidates[self.config.student_model]

        results = []

        for sr_candidate in sr_candidates:
            model = copy.deepcopy(module).to(device)

            # >>>>> extract train samples >>>>>
            num_train = int(math.floor(num_indices * sr_candidate))
            sampled_indices = []
            cnt = 0
            step = int(math.floor(num_indices / num_train))
            for offset in range(0, num_indices, step):
                if cnt == num_train:
                    break
                sampled_indices.append(indices_to_micro_profile[offset])
                cnt += 1
            # <<<<< extract train samples <<<<<

            num_sampled_indices = len(sampled_indices)    
            num_profile = int(math.floor(PROFILE_SCALE * num_train))
            profile_indices = []
            cnt = 0
            step = int(math.floor(num_sampled_indices / num_profile))
            for offset in range(0, num_sampled_indices, step):
                if cnt == num_profile:
                    break
                profile_indices.append(sampled_indices[offset])
                cnt += 1

            num_profile_indices = len(profile_indices)
            num_train_imgs = int(num_profile_indices * 0.8)
            train_indices, valid_indices = split_indices(profile_indices, num_train_imgs)

            # dataset
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            # train_dataset = TrainSampleDataset(ori_dataset=entire_dataset,
            #                                    indices=train_indices)
            entire_dataset.transform = train_transform
            train_loader = DataLoader(dataset=entire_dataset,
                                      batch_size=16,
                                      sampler=CustomSampler(train_indices, shuffle=True),
                                      num_workers=self.config.num_workers,
                                      pin_memory=True,
                                      drop_last=False)
            
            valid_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
            # valid_dataset = TrainSampleDataset(ori_dataset=entire_dataset,
            #                                    indices=valid_indices)
            entire_dataset.transform = valid_transform
            valid_loader = DataLoader(dataset=entire_dataset,
                                      batch_size=16,
                                      sampler=CustomSampler(valid_indices),
                                      num_workers=self.config.num_workers,
                                      pin_memory=True,
                                      drop_last=False)
            #

            # >>>>> before micro training >>>>>
            before_acc_tracker = ClassificationAccuracyTracker()
            self.precision_changer.change_precision(model, ModelPrecision.MX6)
            model.eval()
            with torch.no_grad():
                for inputs, targets, _ in tqdm(valid_loader,
                                               desc=f"before valid with sr: {sr_candidate*100:.1f}%",
                                               unit=" batches"):
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs)

                    top1_acc, _, _ = calculate_accuracy(outputs, targets, topk_list=[1])
                    top1_acc = top1_acc[0]
                
                    before_acc_tracker.update(top1_acc, 0, inputs.shape[0])
            before_avg_acc1 = before_acc_tracker.avg_acc1
            # <<<<< before micro training <<<<<
            
            # >>>>> micro training >>>>>
            criterion = torch.nn.CrossEntropyLoss().to(device=device)
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=self.config.lr,
                                        momentum=self.config.momentum,
                                        weight_decay=self.config.weight_decay)

            self.precision_changer.change_precision(model, ModelPrecision.MX9)
            model.train()
            for e in tqdm(range(self.micro_profile_epoch),
                          desc=f"micro training with sr: {sr_candidate*100:.1f}%",
                          unit=f" epochs"):
                for inputs, targets, _ in train_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs)

                    loss = criterion(outputs, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            # <<<<< micro training <<<<<

            # >>>>> after micro training >>>>>
            after_acc_tracker = ClassificationAccuracyTracker()
            self.precision_changer.change_precision(model, ModelPrecision.MX6)
            model.eval()
            with torch.no_grad():
                for inputs, targets, _ in tqdm(valid_loader,
                                               desc=f"after valid with sr: {sr_candidate*100:.1f}%",
                                               unit=" batches"):
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs)

                    top1_acc, _, _ = calculate_accuracy(outputs, targets, topk_list=[1])
                    top1_acc = top1_acc[0]
                
                    after_acc_tracker.update(top1_acc, 0, inputs.shape[0])
            after_avg_acc1 = after_acc_tracker.avg_acc1
            # <<<<< before micro training <<<<<

            print(f"sr: {sr_candidate*100:.1f}%, "
                  f"before acc.: {before_avg_acc1:.1f}% -> "
                  f"after acc.: {after_avg_acc1:.1f}%")

            results.append({
                "sr": sr_candidate,
                "before_acc": before_avg_acc1,
                "after_acc": after_avg_acc1,
                "train_indices": sampled_indices
            })

        return results
    
    def __check_config(self,
                       profile_result: Dict[str, Any],
                       epoch: int):
        # sr = profile_result["sr"]
        sr = self.label_srs[self.config.student_model]
        before_acc = profile_result["before_acc"] / 100.
        after_acc = profile_result["after_acc"] / 100.
        train_indices = profile_result["train_indices"]
        num_train_data = len(train_indices)

        # >>>>> check time >>>>>
        iter_T_p1 = self.calculate_iter(m=self.m_T_p1,
                                        f=self.total_row - self.static_f_I_p1,
                                        batch_size=self.batch_size,
                                        type=TRAIN)
        
        iter_L_p2 = self.calculate_iter(m=self.m_L_p2,
                                        f=self.total_row - self.static_f_I_p2,
                                        batch_size=1,
                                        type=LABEL)

        window_time = self.config.window_time
        fps = self.config.fps
        num_imgs_per_window = window_time * fps

        num_imgs_to_label = int(math.floor(sr * num_imgs_per_window))
        p2 = iter_L_p2 * num_imgs_to_label

        p1 = window_time - p2
        retrain_iters = math.ceil(num_train_data / self.batch_size) * epoch
        retrain_time = iter_T_p1 * retrain_iters

        if p1 + p2 > window_time or retrain_time > p1:
            return None
        # <<<<< check time <<<<<

        expected_acc = get_expected_accuracy(before_train_acc=before_acc,
                                             after_train_acc=after_acc,
                                             microprofile_epoch=self.micro_profile_epoch,
                                             epoch=epoch)
        
        result = {
            "expected_acc": expected_acc,
            "sr": sr,
            "epoch": epoch,
            "p1": p1,
            "p2": p2,
            "iter_T_p1": iter_T_p1,
            "iter_L_p2": iter_L_p2,
            "before_acc": profile_result["before_acc"],
            "after_acc": profile_result["after_acc"],
            "train_indices": train_indices
        }
        
        return result

    def profile(self,
                module: nn.Module,
                entire_dataset: Dataset,
                indices_to_micro_profile: List[int]) -> Tuple[ProfileResult, List[int]]:
        if self.is_first:
            self.is_first = False
            return self.generate_default_profile_result(), []
        
        # micro profile
        profile_results = self.__micro_profile(module=module,
                                               entire_dataset=entire_dataset,
                                               indices_to_micro_profile=indices_to_micro_profile)
        
        # find best config
        configs = []
        for profile_result in profile_results:
            for epoch in EPOCH_CANDIDATES:
                config = self.__check_config(profile_result, epoch)
                if config is not None:
                    configs.append(config)
        configs = sorted(configs, key=lambda x: x["expected_acc"], reverse=True)
        best_config = configs[0]

        p1 = best_config["p1"]
        p2 = best_config["p1"]
        sr = best_config["sr"]
        epoch = best_config["epoch"]
        iter_T_p1 = best_config["iter_T_p1"]
        iter_L_p2 = best_config["iter_L_p2"]
        train_indices = best_config["train_indices"]

        # >>>>> check frame drop >>>>>
        iter_I_p1 = self.calculate_iter(m=self.static_m_I_p1,
                                        f=self.static_f_I_p1,
                                        batch_size=1,
                                        type=INFERENCE)
        drop_ratio_p1 = np.max([0., 1. - (1. / (self.fps * iter_I_p1))])
        if drop_ratio_p1 != 0:
            # print("invalid config, inference drop at phase 1")
            return None

        iter_I_p2 = self.calculate_iter(m=self.static_m_I_p2,
                                        f=self.static_f_I_p2,
                                        batch_size=1,
                                        type=INFERENCE)
        drop_ratio_p2 = np.max([0., 1. - (1. / (self.fps * iter_I_p2))])
        if drop_ratio_p2 != 0:
            # print("invalid config, inference drop at phase 2")
            return None
        # <<<<< check frame drop <<<<<
        
        return ProfileResult(p1_time=p1,
                             m_T_p1=self.static_m_T_p1,
                             m_I_p1=self.static_m_I_p1,
                             f_I_p1=self.static_f_I_p1,
                             iter_I_p1=iter_I_p1,
                             iter_T_p1=iter_T_p1,
                             p2_time=p2,
                             m_L_p2=self.static_m_L_p2,
                             m_I_p2=self.static_m_I_p2,
                             f_I_p2=self.static_f_I_p2,
                             iter_I_p2=iter_I_p2,
                             iter_L_p2=iter_L_p2,
                             epochs=epoch,
                             desired_sampling_rate=self.label_srs[self.config.student_model],
                             train_sampling_rate=self.label_srs[self.config.student_model],
                             fixed_valid_sampling_rate=0.), train_indices

    def generate_default_profile_result(self) -> ProfileResult:
        # find case of no training cost and labeling maximum desired sampling rate
        results = []

        for m_I_p2 in self.M_I_p2:
            for f_I_p2 in self.F_I_p2:
                for m_I_p1 in self.M_I_p1:
                    value = self.__obj_func_without_training(m_I_p1=m_I_p1,
                                                             m_I_p2=m_I_p2,
                                                             f_I_p1=self.total_row,
                                                             f_I_p2=f_I_p2,
                                                             remain_time=self.config.window_time)
            
                    if value is not None:
                        results.append(value)

        results = sorted(results, key=lambda x: x.p1_time)

        if len(results) > 0:
            return results[0]
        
        raise ValueError(f"cannot start Phoenix profiler")
    
    def __obj_func_without_training(self,
                                    m_I_p1: int,
                                    m_I_p2: int,
                                    f_I_p1: int,
                                    f_I_p2: int,
                                    remain_time: int):
        iter_I_p1 = self.calculate_iter(m=m_I_p1,
                                        f=f_I_p1,
                                        batch_size=1,
                                        type=INFERENCE)
        drop_ratio_p1 = np.max([0., 1. - (1. / (self.fps * iter_I_p1))])
        
        iter_I_p2 = self.calculate_iter(m=m_I_p2,
                                        f=f_I_p2,
                                        batch_size=1,
                                        type=INFERENCE)
        drop_ratio_p2 = np.max([0., 1. - (1. / (self.fps * iter_I_p2))])

        if drop_ratio_p1 != 0 or drop_ratio_p2 != 0:
            return None

        num_total_imgs = self.config.fps * self.config.window_time
        sr_to_sample = self.label_srs[self.config.student_model] # self.config.initial_sampling_rate
        num_imgs_to_sample = int(math.floor(num_total_imgs * sr_to_sample))

        iter_L_p2 = self.calculate_iter(m=self.m_L_p2,
                                        f=self.total_row - f_I_p2,
                                        batch_size=1,
                                        type=LABEL)
        
        p2 = iter_L_p2 * num_imgs_to_sample
        p1 = remain_time - p2

        if p1 <= 0 or p1 + p2 > remain_time:
            return None
        
        return ProfileResult(p1_time=p1,
                             m_T_p1=ModelPrecision.MX9,
                             m_I_p1=m_I_p1,
                             f_I_p1=f_I_p1,
                             iter_I_p1=iter_I_p1,
                             iter_T_p1=0.,
                             p2_time=p2,
                             m_L_p2=ModelPrecision.MX6,
                             m_I_p2=m_I_p2,
                             f_I_p2=f_I_p2,
                             iter_I_p2=iter_I_p2,
                             iter_L_p2=iter_L_p2,
                             epochs=0,
                             desired_sampling_rate=sr_to_sample,
                             train_sampling_rate=0.,
                             fixed_valid_sampling_rate=0.)


if __name__ == "__main__":
    pass