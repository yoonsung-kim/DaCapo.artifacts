import gc
import csv
import json
import math
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import gaussian_blur
from config import Config
from model import ModelGenertor
from reproduce import set_reproducibility
from accuracy import ClassificationAccuracyTracker, calculate_accuracy
from dataset_util import generate_windows, EntireDataset, CustomSampler


WARM_UP_TIME_SECONDS = 480

SAMPLING_RATE_PER_MODEL = {
    "wide_resnet50_2": 0.19,
    "wide_resnet101_2": 0.04,
    "vit_b_16": 0.17
}

parser = argparse.ArgumentParser(description="Additional baseline experiment")
parser.add_argument("--config-path", type=str, help="path of configuration .json file")

class AdditionalBaseline:
    def __init__(self, config: Config) -> None:
        self.config = config

        set_reproducibility(self.config.seed)

        self.student_model = ModelGenertor.generate(name=self.config.student_model,
                                                    num_classes=self.config.num_classes,
                                                    weight_path=self.config.student_weight)
        self.student_model = self.student_model.to(self.config.train_device)
        
        self.teacher_model = ModelGenertor.generate(name=self.config.teacher_model,
                                                    num_classes=self.config.num_classes,
                                                    weight_path=self.config.teacher_weight)
        self.teacher_model = self.teacher_model.to(self.config.label_device)

        self.entire_dataset: Dataset = None
        self.windows: List[Dataset] = []
        self.num_windows: int = None


        self.retrained_accuracies: List[float] = []

        self.infer_accuracy_per_window: List[float] = []
        self.corrects = []

        self.prev_indices = []
        self.imgs_per_window = self.config.fps * self.config.window_time

        self.epoch_min = 1
        self.epoch_max = 30
        self.num_train_imgs_min = int(math.floor(self.config.fps * self.config.window_time * 0.05))

        # init log files
        self.output_root = Path(self.config.output_root)

        self.acc_log_path = self.output_root / "result.csv"
        with open(self.acc_log_path, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["window index", "accuracy"])

        self.num_samples_log_path = self.output_root / "num_samples_per_windows.csv"
        with open(self.num_samples_log_path, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["window index", "number of samples"])

        self.train_log_path = self.output_root / "train_log.csv"
        with open(self.train_log_path, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "window index",
                "epoch",
                "num of samples",
                "batch size"
                "iteration count"
            ])

        self.accumulated_window_time = 0 

    def run(self):
        self.entire_dataset = EntireDataset(scenario_dir=Path(self.config.scenario_path),
                                            transform=None)
        self.windows = generate_windows(scenario_path=Path(self.config.scenario_path),
                                        fps=self.config.fps,
                                        window_time=self.config.window_time)
        self.num_windows = len(self.windows)
        print(f"# of windows: {self.num_windows}")

        for window_idx in range(self.num_windows):
            self.__run(window_idx=window_idx)

        correct_log_path = self.output_root / "result.json"
        with open(correct_log_path, "w") as f:
            json.dump(self.corrects, f, indent=4)

    def __run(self, window_idx: int):
        self.accumulated_window_time += self.config.window_time

        if self.accumulated_window_time == WARM_UP_TIME_SECONDS:
            self.retrained_accuracies = []
        
        torch.cuda.synchronize()
        start_time = time.time_ns()

        print(f"START WINDOW {window_idx+1}/{self.num_windows}")

        # inference
        curr_window = self.windows[window_idx]

        infer_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        curr_window.transform = infer_transform
        infer_loader = DataLoader(dataset=curr_window,
                                  batch_size=self.config.infer_batch_size,
                                  num_workers=self.config.num_workers,
                                  drop_last=False,
                                  shuffle=False,
                                  pin_memory=True)
        
        infer_acc_tracker = ClassificationAccuracyTracker()

        self.student_model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(infer_loader,
                                        desc=f"infer at window #{window_idx}",
                                        unit=" images"):
                inputs = inputs.to(self.config.infer_device, non_blocking=True)
                targets = targets.to(self.config.infer_device, non_blocking=True)

                outputs = self.student_model(inputs)

                top1_acc, top1_corrects = calculate_accuracy(outputs, targets, topk_list=[1])
                top1_acc = top1_acc[0]
                top1_corrects = top1_corrects.cpu().numpy().astype(int).tolist()
                
                infer_acc_tracker.update(top1_acc, 0, inputs.shape[0])
                infer_acc_tracker.corrects.extend(top1_corrects)

        # log inference results
        print(f"accuracy at window #{window_idx}: {infer_acc_tracker.avg_acc1:.1f}%")
        self.infer_accuracy_per_window.append(infer_acc_tracker.avg_acc1)
        self.corrects.append(infer_acc_tracker.corrects)

        with open(self.acc_log_path, "a") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([window_idx, infer_acc_tracker.avg_acc1])

        torch.cuda.synchronize()
        end_time = time.time_ns()
        infer_exec_time = (end_time - start_time) / 1_000_000_000

        # key frame extractor
        indices_to_label = self.__extract_key_frames(window=curr_window)

        with open(self.num_samples_log_path, "a") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([window_idx, len(indices_to_label)])

        torch.cuda.synchronize()
        start_time = time.time_ns()

        # label
        self.teacher_model.eval()
        with torch.no_grad():
            label_loader = DataLoader(dataset=curr_window,
                                      batch_size=self.config.infer_batch_size,
                                      num_workers=self.config.num_workers,
                                      sampler=CustomSampler(indices=indices_to_label),
                                      drop_last=False,
                                      pin_memory=True)
            for inputs, _ in tqdm(label_loader,
                                    desc=f"label at window #{window_idx}",
                                    unit=" sample"):
                inputs = inputs.to(self.config.label_device, non_blocking=True)
                outputs = self.teacher_model(inputs)

        torch.cuda.synchronize()
        end_time = time.time_ns()
        label_exec_time = (end_time - start_time) / 1_000_000_000

        print(f"infer time at window #{window_idx}: {infer_exec_time:.1f} seconds")
        print(f"label time at window #{window_idx}: {label_exec_time:.1f} seconds")
        print(f"remain time at window #{window_idx}: {self.config.window_time - (infer_exec_time + label_exec_time):.1f} seconds")

        # calculate accuracy matched to labeled data
        labeled_corrects = []
        for labeled_index in indices_to_label:
            labeled_corrects.append(infer_acc_tracker.corrects[labeled_index])
        labeled_accuracy = np.sum(labeled_corrects) / len(labeled_corrects) * 100.

        labeled_accuracy = np.sum(labeled_corrects) / len(labeled_corrects) * 100.

        # trigger controller
        should_retrain = self.__trigger_retrain(curr_acc=labeled_accuracy)

        # log inference information
        self.retrained_accuracies.append(labeled_accuracy)

        # retraining manager
        if should_retrain or self.accumulated_window_time < WARM_UP_TIME_SECONDS:
            # find configuration
            epoch, num_imgs = self.__find_configuration(curr_accuracy=labeled_accuracy,
                                                        labeled_indices=indices_to_label)

            iter_cnt = 0
            if infer_exec_time + label_exec_time < self.config.window_time:
                # retrain
                iter_cnt = self.__retrain(window_idx=window_idx,
                                          start_time=start_time,
                                          epoch=epoch,
                                          num_imgs=num_imgs,
                                          executed_time = infer_exec_time + label_exec_time,
                                          labeled_indices=indices_to_label)
            else:
                print("no time to train")
                
            with open(self.train_log_path, "a") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([
                    window_idx,
                    epoch,
                    num_imgs,
                    self.config.train_batch_size,
                    iter_cnt
                ])

        global_indices = []
        for sample_idx in indices_to_label:
            global_indices.append(sample_idx * self.imgs_per_window)
        self.prev_indices.extend(global_indices)

    def __extract_key_frames(self, window: Dataset) -> List[int]:
        beta_threshold = 0.4
        filter_threshold = 0.5
        indices = []

        for i in tqdm(range(0, len(window) - 1)):
            img_0, _ = window[i]
            img_1, _ = window[i+1]

            diff = torch.clamp(img_1 - img_0, min=0.0, max=1.0)
            diff_cvt = 0.299 * diff[0] + 0.587 * diff[1] + 0.114 * diff[2]
            diff_cvt = torch.unsqueeze(diff_cvt, dim=0)
            diff_cvt = gaussian_blur(diff_cvt, kernel_size=[3,3])
            diff_cvt = torch.squeeze(diff_cvt, dim=0)
            diff_cvt = (diff_cvt >= filter_threshold).float()

            value = torch.count_nonzero(diff_cvt) / torch.numel(diff_cvt)

            if value > beta_threshold:
                indices.append(i)

        # sub-sampling awaring labeling cost
        sampling_rate = SAMPLING_RATE_PER_MODEL[self.config.teacher_model]
        num_imgs_per_window = self.config.window_time * self.config.fps
        num_to_label = int(math.floor(sampling_rate * num_imgs_per_window))
        num_indices = len(indices)

        if num_indices == num_to_label:
            print(f"samples to be labeled: {num_indices}/{len(window)}")
            return indices
        
        sampled_indices = []
        cnt = 0
        step = int(math.floor(num_indices / num_to_label))

        if step < 1:
            return indices

        print(f"{num_indices}, {num_to_label}, {step}")
        for offset in range(0, num_indices, step):
            if cnt == num_to_label:
                break
            sampled_indices.append(indices[offset])
        print(f"samples to be labeled: {len(sampled_indices)}/{len(window)}")

        return sampled_indices

    def __trigger_retrain(self, curr_acc: float) -> bool:
        def exp_decay(x):
            return np.exp(-x)
        prev_accs = np.array(self.retrained_accuracies)
        
        scale = exp_decay(np.arange(len(prev_accs)))
        scale = scale / np.sum(scale)

        weighted_accs = scale * prev_accs[::-1]
        sum_weighted_accs = np.sum(weighted_accs)
        std_acc = np.std(self.retrained_accuracies) if len(self.retrained_accuracies) > 1 else 0

        # [MM'23 version] need to retrain
        if sum_weighted_accs - std_acc > curr_acc:
            print(f"need to retrain: {sum_weighted_accs:.1f}% - {std_acc:.1f}% > {curr_acc:.1f}%")
            return True
        else:
            print(f"no need to retrain: {sum_weighted_accs:.1f}% - {std_acc:.1f}% <= {curr_acc:.1f}%")
            return False

    def __find_configuration(self,
                             curr_accuracy: float,
                             labeled_indices: List[int]) -> Tuple[int]:
        def generate_new_config(old_epoch: int,
                                old_num_imgs: int,
                                num_labeled_imgs: int) -> Tuple[int]:
            epoch_min = self.epoch_min
            epoch_max = self.epoch_max
            num_imgs_min = self.num_train_imgs_min

            epoch_to_add = np.random.randint(low=-1*epoch_min,
                                             high=epoch_min+1)

            new_epoch = max(min(old_epoch+epoch_to_add,
                                epoch_max),
                            epoch_min)
            new_num_imgs = num_labeled_imgs

            return new_epoch, new_num_imgs
        
        def get_utility(epoch: int,
                        num_imgs: int,
                        num_labeled_imgs: int,
                        curr_accuracy: float) -> float:
            train_iter_time = self.config.train_iter_time
            label_iter_time = self.config.label_iter_time

            urgency_degree = num_labeled_imgs / (self.config.fps * self.config.window_time)
            time_cost = (label_iter_time * num_labeled_imgs) + (epoch * num_imgs * train_iter_time)
            return max(0, curr_accuracy - (urgency_degree * time_cost))
        
        i = 0 
        j = 0
        max_iter = 100
        max_fail = 5

        num_labeled_imgs = len(labeled_indices)

        # initial config
        best_epoch = int(math.floor(self.epoch_max / 2.))
        best_num_imgs = num_labeled_imgs
        best_utility_score = get_utility(epoch=best_epoch,
                                         num_imgs=best_num_imgs,
                                         num_labeled_imgs=num_labeled_imgs,
                                         curr_accuracy=curr_accuracy)

        while i < max_iter and j < max_fail:
            # generate new config
            new_epoch, new_num_imgs = generate_new_config(old_epoch=best_epoch,
                                                          old_num_imgs=best_num_imgs,
                                                          num_labeled_imgs=num_labeled_imgs)
            
            new_utility_score = get_utility(epoch=new_epoch,
                                            num_imgs=new_num_imgs,
                                            num_labeled_imgs=num_labeled_imgs,
                                            curr_accuracy=curr_accuracy)
            
            diff = new_utility_score - best_utility_score
            rand_value = np.random.rand()

            if diff > 0 or np.exp(diff / (new_utility_score + 1e-5)) > rand_value:
                best_epoch = new_epoch
                best_num_imgs = new_num_imgs
                best_utility_score = new_utility_score
                j = 0
            else:
                j += 1
            i += 1

        return new_epoch, new_num_imgs

    def __retrain(self,
                  window_idx: int,
                  start_time: int,
                  epoch: int,
                  num_imgs: int,
                  executed_time: float,
                  labeled_indices: List[int]) -> int:
        remain_time = self.config.window_time - executed_time

        torch.cuda.synchronize()
        start_time = time.time_ns()

        num_labeled_indices = len(labeled_indices)
        if num_labeled_indices == 0:
            return 0 # if no labeled data, return iteration count as 0

        if num_labeled_indices == num_imgs:
            indices = labeled_indices
        else:
            indices = []
            cnt = 0
            step = int(math.floor(num_labeled_indices / num_imgs))
            print(f"range: {0}, {num_labeled_indices}, {step} ({num_imgs})")
            for offset in range(0, num_labeled_indices, step):
                if cnt == num_imgs:
                    break
                indices.append(labeled_indices[offset])
                cnt += 1

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        criterion = torch.nn.CrossEntropyLoss().to(self.config.train_device)
        optimizer = torch.optim.SGD(self.student_model.parameters(),
                                    lr=self.config.lr,
                                    momentum=self.config.momentum,
                                    weight_decay=self.config.weight_decay)

        iter_cnt = 0
        should_stop = False

        self.student_model.train()
        for e in tqdm(range(epoch),
                      desc=f"retrain at window #{window_idx}",
                      unit=" epochs"):
            train_dataset = self.windows[window_idx]
            train_dataset.transform = train_transform
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=self.config.train_batch_size,
                                      sampler=CustomSampler(indices=indices, shuffle=True),
                                      num_workers=self.config.num_workers,
                                      drop_last=False,
                                      pin_memory=True)
            for inputs, targets in train_loader:
                inputs = inputs.to(self.config.train_device, non_blocking=True)
                targets = targets.to(self.config.train_device, non_blocking=True)

                outputs = self.student_model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                iter_cnt += 1

                torch.cuda.synchronize()
                end_time = time.time_ns()

                exec_time = (end_time - start_time) / 1_000_000_000
                if exec_time >= remain_time:
                    print(f"time out for window #{window_idx}: "
                          # f"{exec_time:.1f} > {self.config.window_time:.1f} seconds, "
                          f"{exec_time:.1f} > {remain_time:.1f} (remain time) seconds, "
                          f"exit retraining")
                    should_stop = True
                    break

            if should_stop:
                break
        
        return iter_cnt


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    args = parser.parse_args()
    additional_baseline = AdditionalBaseline(config=Config(config_file_path=args.config_path))

    start_time = time.time_ns()
    additional_baseline.run()
    end_time = time.time_ns()
    exec_time = (end_time - start_time) / 1_000_000_000
    print(f"EXPERIMENT EXECUTION TIME: {exec_time / 60:.1f} minutes")