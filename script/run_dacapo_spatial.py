import csv
import json
import math
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from util.reproduce import set_reproducibility
from util.accuracy import ClassificationAccuracyTracker
from emulator.config import Config
from emulator.logger import Logger
from emulator.profiler import Profiler
from emulator.model import ModelPrecision
from emulator.statistic import WindowStatistic
from emulator.model_factory import ModelFactory
from emulator.profiler_factory import ProfilerFactory
from emulator.dataset import split_dataset_indices_by_time, MystiqueDataset, TrainSampleDataset, CustomSampler


parser = argparse.ArgumentParser(description="Continual learning emluator")
parser.add_argument("--config-path", type=str, help="path of emulator configuration .json file")


class Emulator:
    def __init__(self, config: Config):
        self.config: Config = config

        set_reproducibility(self.config.seed)

        if self.config.cl_type != "STATIC":
            raise ValueError(f"continual learning type must be STATIC, not {self.config.cl_type}")
            
        self.profiler: Profiler = ProfilerFactory.generate_profiler(config=self.config)

        #TODO: generate entire dataset first and split by time
        self.student_model = ModelFactory.generate_model(config=self.config,
                                                         task_type=self.config.task_type,
                                                         name=self.config.student_model,
                                                         precision=ModelPrecision.MX9,
                                                         batch_size=self.config.infer_batch_size,
                                                         num_classes=self.config.num_classes,
                                                         device=self.config.infer_device,
                                                         freeze=False,
                                                         weight_path=self.config.student_weight)

        self.train_dataset_from_prev_window: List[TrainSampleDataset] = [None,]

        self.statistics_from_prev_window: List[WindowStatistic] = [None,]

        self.logger = Logger(config=self.config, name=self.config.cl_type)

        # >>>>> log window dist >>>>>
        with open(f"{self.config.output_root}/window_dist_log.csv", "w") as f:
            csv_writer = csv.writer(f)
            header = [
                "Window index",
            ]
            for l in range(self.config.num_classes):
                header.append(f"Label {l} # of images")
                header.append(f"Label {l} data ratio")

            csv_writer.writerow(header)

        with open(f"{self.config.output_root}/train_sample_dist_log.csv", "w") as f:
            csv_writer = csv.writer(f)
            header = [
                "Window index",
                "Sampling rate",
                "# of total images",
                "# of sampled images",
            ]
            for l in range(self.config.num_classes):
                header.append(f"Label #{l} # of images")
                header.append(f"Label #{l} ratio")

            csv_writer.writerow(header)
        # <<<<< log window dist <<<<<

    def run(self):
        self.entire_dataset = MystiqueDataset(scenario_dir=Path(self.config.scenario_path),
                                              transform=None)
        #TODO: get dataset from WW
        self.latency_table = self.entire_dataset.generate_frame_latency_table(fps=self.config.fps)

        start_idx = 0
        start_time = 0
        duration_time = self.config.window_time

        self.window_index_slices = []

        while start_idx != -1:
            index_slice, \
            next_start_idx, \
            next_start_time, \
            time = split_dataset_indices_by_time(frame_latency_table=self.latency_table,
                                                 start_index=start_idx,
                                                 start_time=start_time,
                                                 duration_time=duration_time)
            #TODO: log in here
            print(f"window idx: {len(self.window_index_slices)}")
            print(f"time: {time:.1f} seconds")
            print(f"start idx: {start_idx} (time: {start_time:.1f} seconds)")
            print(f"index slice:{index_slice} (# of images: {index_slice[1] - index_slice[0] + 1})")

            start_idx = next_start_idx
            start_time = next_start_time
            self.window_index_slices.append(index_slice)
    
        for window_idx in range(len(self.window_index_slices)):
            print(f"window #{window_idx}")
            self.__run(window_idx=window_idx,
                       window_index_slice=self.window_index_slices[window_idx])
            
        json.dump(self.logger.corrects, open(self.logger.corrects_path, "w"), indent=4)

    def __run(self,
              window_idx: int,
              window_index_slice: Tuple[int, int]) -> WindowStatistic:
        # >>>>> log >>>>>
        sample_cnt_per_label = {i: 0 for i in range(self.config.num_classes)}

        data_loader = DataLoader(dataset=self.entire_dataset,
                                 sampler=CustomSampler(indices=list(range(window_index_slice[0],
                                                                          window_index_slice[1] + 1))),
                                 shuffle=False,
                                 pin_memory=True,
                                 batch_size=self.config.infer_batch_size,
                                 num_workers=self.config.num_workers)

        for inputs, targets, _ in tqdm(data_loader,
                                    desc="Checking sampled train dataset",
                                    unit=" batches"):
            for b in range(inputs.shape[0]):
                label = targets[b].item()
                sample_cnt_per_label[label] += 1

        num_imgs = window_index_slice[1] - window_index_slice[0] + 1

        with open(f"{self.config.output_root}/window_dist_log.csv", "a") as f:
            csv_writer = csv.writer(f)

            row = [window_idx]
            for l in sample_cnt_per_label.keys():
                print(f"label #{l}: {sample_cnt_per_label[l]} images ({sample_cnt_per_label[l] / num_imgs * 100.:.1f}%)")
                row.append(sample_cnt_per_label[l])
                row.append(sample_cnt_per_label[l] / num_imgs * 100.)

            csv_writer.writerow(row)
        # <<<<< log <<<<< 
        prev_stat = self.statistics_from_prev_window[window_idx]

        stat = WindowStatistic(window_index=window_idx,
                               config=self.config,
                               profile_result=None)

        window_accuracy_tracker = ClassificationAccuracyTracker()
        p1_accuracy_tracker = ClassificationAccuracyTracker()
        p2_accuracy_tracker = ClassificationAccuracyTracker()

        profile_result = self.profiler.profile(prev_statistic=None,
                                               max_train_sampling_rate=None,
                                               remain_time=None)
        print(f">>>>> profiled result >>>>>")
        print(profile_result)
        print(f"<<<<< profiled result <<<<<")
        
        p1_index_slice, p2_index_slice = self.__split_indice_by_phase(p1_time=profile_result.p1_time,
                                                                      window_index_slice=window_index_slice)
        p1_infer_num_imgs = p1_index_slice[1] - p1_index_slice[0] + 1
        p2_infer_num_imgs = p2_index_slice[1] - p2_index_slice[0] + 1

        print(f"p1: {p1_infer_num_imgs} images, p2: {p2_infer_num_imgs} images")

        # >>> phase 1
        # retraining mode: we guarantee training job within the window
        self.student_model.change_precision(precision=profile_result.m_I_p1)
        print(f"change precision to {profile_result.m_I_p1} bit for phase #1 inference")
        self.student_model.infer(phase=1,
                                 dataset=self.entire_dataset,
                                 index_slice=p1_index_slice,
                                 phase_accuracy_tracker=p1_accuracy_tracker,
                                 phase_loss_tracker=None,
                                 window_accuracy_tracker=window_accuracy_tracker)
        p1_acc = p1_accuracy_tracker.avg_acc1
        print(f"exec time: {profile_result.p1_time:.1f} seconds, accuracy: {p1_acc:.1f}%")
        
            
        self.student_model.change_precision(precision=ModelPrecision.MX9)
        print(f"change precision to {ModelPrecision.MX9} bit for phase #1 training")
        train_dataset = self.train_dataset_from_prev_window[window_idx]
        _ = self.student_model.train(train_dataset=train_dataset,
                                     valid_dataset=None,
                                     epochs=profile_result.epochs)
            
        stat.epochs = profile_result.epochs
        stat.train_sampling_rate = profile_result.train_sampling_rate
        stat.fixed_valid_sampling_rate = profile_result.fixed_valid_sampling_rate
        stat.desired_sampling_rate = profile_result.desired_sampling_rate

        stat.p1_time = profile_result.p1_time
        stat.m_I_p1 = profile_result.m_I_p1
        stat.f_I_p1 = profile_result.f_I_p1
        stat.p1_num_imgs = p1_infer_num_imgs
        
        stat.p2_time = profile_result.p2_time
        stat.m_L_p2 = profile_result.m_L_p2
        stat.m_I_p2 = profile_result.m_I_p2
        stat.f_I_p2 = profile_result.f_I_p2
        stat.p2_num_imgs = p2_infer_num_imgs

        stat.p1_acc = p1_acc
        # print(f"exec time: {stat.p1_time:.1f} seconds, accuracy: {stat.p1_acc:.1f}%")
        # <<< phase 1

        # >>> phase 2
        # labeling: we guarantee data labeling job within the window
        # desired_sampling_rate = profile_result.desired_sampling_rate

        # if self.config.cl_type != "NONE":
        #     train_dataset = self.__sample_data_to_train(window_idx=window_idx,
        #                                                 entire_dataset=self.entire_dataset,
        #                                                 index_slice=window_index_slice,
        #                                                 desired_sampling_rate=desired_sampling_rate)
        #     self.train_dataset_from_prev_window.append(train_dataset)
        # else:
        #     print("no labeling")
        #     self.train_dataset_from_prev_window.append(None)
        
        # inference
        self.student_model.change_precision(precision=profile_result.m_I_p2)
        print(f"change precision to {profile_result.m_I_p2} bit for phase #2 inference")
        print(f"# of images: {p2_infer_num_imgs}")
        self.student_model.infer(phase=2,
                                 dataset=self.entire_dataset,
                                 index_slice=p2_index_slice,
                                 phase_accuracy_tracker=p2_accuracy_tracker,
                                 phase_loss_tracker=None,
                                 window_accuracy_tracker=window_accuracy_tracker)

        stat.p2_acc = p2_accuracy_tracker.avg_acc1
        print(f"exec time: {stat.p1_time:.1f} seconds, accuracy: {stat.p2_acc:.1f}%")
        # <<< phase 2

        stat.window_acc = window_accuracy_tracker.avg_acc1

        # labeling: we guarantee data labeling job within the window
        desired_sampling_rate = profile_result.desired_sampling_rate

        confidences = []
        confidences.extend(p1_accuracy_tracker.confidences)
        confidences.extend(p2_accuracy_tracker.confidences)

        if self.config.cl_type != "NONE":
            train_dataset = self.__sample_data_to_train(window_idx=window_idx,
                                                        entire_dataset=self.entire_dataset,
                                                        index_slice=window_index_slice,
                                                        confidences=confidences,
                                                        desired_sampling_rate=desired_sampling_rate)
            self.train_dataset_from_prev_window.append(train_dataset)
        else:
            print("no labeling")
            self.train_dataset_from_prev_window.append(None)
        
        # >>> summary
        print(f"time -> p1: {stat.p1_time:.1f} seconds, p2: {stat.p2_time:.1f} seconds")
        print(f"accs -> p1: {stat.p1_acc:.1f}%, p2: {stat.p2_acc:.1f}%")
        if prev_stat is not None:
            print(f"pred accuracy: {prev_stat.p2_measured_metric:.1f}%")
        print(f"profiled loss: {stat.pred_acc:.1f}%")
        # <<< summary
        # <<< phase 2

        self.logger.write_window_info(stat=stat)

        stat.summary()
        self.statistics_from_prev_window.append(stat)

        self.logger.corrects[window_idx] = window_accuracy_tracker.corrects

    def __split_indice_by_phase(self,
                                p1_time: float,
                                window_index_slice: Tuple[int, int]) -> Tuple[Tuple[int, int]]:
        sub_latency_table = self.latency_table[window_index_slice[0]:window_index_slice[1] + 1]

        print(f"{sub_latency_table[0]:.1f} seconds (idx: {window_index_slice[0]}), {sub_latency_table[-1]:.1f} seconds (idx: {window_index_slice[1]})")

        p1_time = sub_latency_table[0] + p1_time
        for index, value in enumerate(sub_latency_table):
            if value <= p1_time:
                last_index = index

        p1_index_slice = (window_index_slice[0], window_index_slice[0] + last_index)
        p2_index_slice = (window_index_slice[0] + last_index + 1, window_index_slice[1])

        print(f"p1 index slice: {p1_index_slice}")
        print(f"p2 index slice: {p2_index_slice}")

        return p1_index_slice, p2_index_slice
    
    def __sample_data_to_train(self,
                               window_idx: int,
                               entire_dataset: MystiqueDataset,
                               index_slice: Tuple[int, int],
                               confidences: List[float],
                               desired_sampling_rate: float) -> Dataset:
        if self.config.use_active_cl:
            return self.__sample_data_to_train_active_cl(window_idx=window_idx,
                                                         entire_dataset=entire_dataset,
                                                         index_slice=index_slice,
                                                         confidences=confidences,
                                                         desired_sampling_rate=desired_sampling_rate)
        else:
            return self.__sample_data_to_train_uniform(window_idx=window_idx,
                                                       entire_dataset=entire_dataset,
                                                       index_slice=index_slice,
                                                       desired_sampling_rate=desired_sampling_rate)
    
    def __sample_data_to_train_uniform(self,
                                       window_idx: int,
                                       entire_dataset: MystiqueDataset,
                                       index_slice: Tuple[int, int],
                                       desired_sampling_rate: float) -> Dataset:
        total_num = index_slice[1] - index_slice[0] + 1
        train_num_imgs = int(math.floor(total_num * desired_sampling_rate))
        step = int(np.floor(total_num / train_num_imgs))

        train_sample_indices = []

        for offset in range(0, total_num, step):
            if len(train_sample_indices) == train_num_imgs:
                break
            
            train_sample_indices.append(index_slice[0] + offset)

        print(f"# of images: {total_num}, sampled # of images: {len(train_sample_indices)} ({len(train_sample_indices) / total_num * 100.:.1f}%)")

        train_dataset = TrainSampleDataset(ori_dataset=entire_dataset,
                                           indices=train_sample_indices)
        
        # >>>>> log >>>>>
        sample_cnt_per_label = {i: 0 for i in range(self.config.num_classes)}

        data_loader = DataLoader(dataset=train_dataset,
                                 shuffle=False,
                                 pin_memory=True,
                                 batch_size=self.config.infer_batch_size,
                                 num_workers=self.config.num_workers)
        
        for inputs, targets, _ in tqdm(data_loader,
                                   desc="Checking sampled train dataset",
                                   unit=" batches"):
            for b in range(inputs.shape[0]):
                label = targets[b].item()
                sample_cnt_per_label[label] += 1
        
        sampled_num = len(train_dataset)

        row = [window_idx, desired_sampling_rate, total_num, sampled_num]
        
        print(f"window #{window_idx} sample {desired_sampling_rate*100.:.1f}% data -> # of {sampled_num} images")
        
        for l in sample_cnt_per_label.keys():
            print(f"label #{l}: {sample_cnt_per_label[l]} images ({sample_cnt_per_label[l] / sampled_num * 100.:.1f}%)")
            row.append(sample_cnt_per_label[l])
            row.append(sample_cnt_per_label[l] / sampled_num * 100.)
        
        with open(f"{self.config.output_root}/train_sample_dist_log.csv", "a") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(row)
        # <<<<< log <<<<<<

        return train_dataset

    def __sample_data_to_train_active_cl(self,
                                         window_idx: int,
                                         entire_dataset: MystiqueDataset,
                                         index_slice: Tuple[int, int],
                                         confidences: List[float],
                                         desired_sampling_rate: float) -> Dataset:
        total_num = index_slice[1] - index_slice[0] + 1
        train_num_imgs = int(math.floor(total_num * desired_sampling_rate))

        # print(f"total num: {total_num}, confidences: {len(confidences)}")
        assert (total_num == len(confidences))

        candidate_samples = []
        for i in range(total_num):
            candidate_samples.append({
                "index": index_slice[0] + i,
                "confidence": confidences[i]
            })
        
        confidence_pivot = 0.5
        candidate_samples = sorted(candidate_samples,
                                   key=lambda x: abs(x["confidence"] - confidence_pivot))
        candidate_samples = candidate_samples[0:train_num_imgs]

        train_sample_indices = []

        for i in range(len(candidate_samples)):
            train_sample_indices.append(candidate_samples[i]["index"])

        print(f"# of images: {total_num}, sampled # of images: {len(train_sample_indices)} ({len(train_sample_indices) / total_num * 100.:.1f}%)")

        train_dataset = TrainSampleDataset(ori_dataset=entire_dataset,
                                           indices=train_sample_indices)
        
        # >>>>> log >>>>>
        sample_cnt_per_label = {i: 0 for i in range(self.config.num_classes)}

        data_loader = DataLoader(dataset=train_dataset,
                                 shuffle=False,
                                 pin_memory=True,
                                 batch_size=self.config.infer_batch_size,
                                 num_workers=self.config.num_workers)
        
        for inputs, targets, _ in tqdm(data_loader,
                                   desc="Checking sampled train dataset",
                                   unit=" batches"):
            for b in range(inputs.shape[0]):
                label = targets[b].item()
                sample_cnt_per_label[label] += 1
        
        sampled_num = len(train_dataset)

        row = [window_idx, desired_sampling_rate, total_num, sampled_num]
        
        print(f"window #{window_idx} sample {desired_sampling_rate*100.:.1f}% data -> # of {sampled_num} images")
        
        for l in sample_cnt_per_label.keys():
            print(f"label #{l}: {sample_cnt_per_label[l]} images ({sample_cnt_per_label[l] / sampled_num * 100.:.1f}%)")
            row.append(sample_cnt_per_label[l])
            row.append(sample_cnt_per_label[l] / sampled_num * 100.)
        
        with open(f"{self.config.output_root}/train_sample_dist_log.csv", "a") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(row)
        # <<<<< log <<<<<<

        return train_dataset


if __name__ == "__main__":
    args = parser.parse_args()
    emulator = Emulator(config=Config(config_file_path=args.config_path))
    emulator.run()