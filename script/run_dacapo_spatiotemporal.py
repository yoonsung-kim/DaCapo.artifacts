import csv
import copy
import json
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import DataLoader
from util.reproduce import set_reproducibility
from util.accuracy import ClassificationAccuracyTracker
from emulator.config import Config
from emulator.logger import Logger
from emulator.model import ModelPrecision
from emulator.statistic import WindowStatistic
from emulator.model_factory import ModelFactory
from emulator.episodic_memory import EpisodicMemory
from emulator.dacapo_profiler import DacapoProfiler, START_LABEL_SR, START_TRAIN_SR, SR_PAIRS
from emulator.dataset import split_dataset_indices_by_time, MystiqueDataset, TrainSampleDataset


parser = argparse.ArgumentParser(description="Continual learning emluator")
parser.add_argument("--config-path", type=str, help="path of emulator configuration .json file")


class Emulator:
    def __init__(self, config: Config):
        self.config: Config = config

        set_reproducibility(self.config.seed)

        if self.config.cl_type != "DACAPO":
            raise ValueError(f"continual learning type must be DACAPO, not {self.config.cl_type}")
            
        self.profiler: DacapoProfiler = DacapoProfiler(config=self.config)

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

        # >>>>> generate episodic memory >>>>>
        self.max_sampling_img_num = SR_PAIRS[0]["label_sr"]
        self.num_data_for_first_label = self.max_sampling_img_num

        self.data_capacity = 1080
        self.sub_label_sr = int(SR_PAIRS[0]["label_sr"]*  0.25)

        self.episodic_memory = EpisodicMemory(num_classes=self.config.num_classes,
                                              # TODO: set to large? -> due to distribution 
                                              # capacity=self.max_sampling_img_num
                                              capacity=self.data_capacity
                                              )
        # <<<<< generate episodic memory <<<<<

        self.total_acc_tracker = ClassificationAccuracyTracker()

        # >>>>> generate empty log files >>>>>
        self.phase_log_path = f"{self.config.output_root}/phase_log.csv"
        with open(self.phase_log_path, "w") as f:
            csv_writer = csv.writer(f)
            header = [
                "Phase index",
                "Phase name",
                "# of images",
                "Executed time",
                "Accuracy on inference",
                "Train: # of images",
                "Valid: # of images",
                "Label: # of images",
                "Buffer count",
                "Buffer capacity"
            ]

            csv_writer.writerow(header)

        with open(f"{self.config.output_root}/train_sample_dist_log.csv", "w") as f:
            csv_writer = csv.writer(f)
            header = [
                "Phase index",
                "# of train samples",
            ]
            for l in range(self.config.num_classes):
                header.append(f"Label #{l} # of images")
                header.append(f"Label #{l} ratio")

            csv_writer.writerow(header)

        with open(f"{self.config.output_root}/valid_sample_dist_log.csv", "w") as f:
            csv_writer = csv.writer(f)
            header = [
                "Phase index",
                "# of valid samples",
            ]
            for l in range(self.config.num_classes):
                header.append(f"Label #{l} # of images")
                header.append(f"Label #{l} ratio")

            csv_writer.writerow(header)

        self.corrects_per_phase = []
        # <<<<< generate empty log files <<<<<

    def run(self):
        self.entire_dataset = MystiqueDataset(scenario_dir=Path(self.config.scenario_path),
                                              transform=None)
        self.latency_table = self.entire_dataset.generate_frame_latency_table(fps=self.config.fps)

        is_done = False
        start_index = 0
        start_time = 0
        train_sr = START_TRAIN_SR
        label_sr = START_LABEL_SR

        self.phase_index = 0

        while not is_done:
            is_done, \
            next_start_idx, \
            next_start_time, \
            time, \
            next_train_sr, \
            next_label_sr = self.__run(start_index=start_index,
                                       start_time=start_time,
                                       train_sr=train_sr,
                                       label_sr=label_sr)

            start_index = next_start_idx
            start_time = next_start_time
            train_sr = next_train_sr
            label_sr = next_label_sr

            self.phase_index += 1

        print(f"total images: {int(np.sum([len(v) for v in self.corrects_per_phase]))}")
        with open(f"{self.config.output_root}/result.json", "w") as f:
            json.dump(self.corrects_per_phase, f, indent=4)

        with open(f"{self.config.output_root}/result.csv", "w") as f:
            all_corrects = []
            for c in self.corrects_per_phase:
                all_corrects.extend(c)

            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "Window index",
                "Accuracy"
            ])

            for w in range(0, len(all_corrects), self.config.window_time * self.config.fps):
                sub_corrects = all_corrects[w:w+(self.config.window_time * self.config.fps)]
                print(len(sub_corrects))
                assert (len(sub_corrects) == self.config.window_time * self.config.fps)
                csv_writer.writerow([
                    int(w / (self.config.window_time * self.config.fps)),
                    np.sum(sub_corrects) / len(sub_corrects) * 100.
                ])

        print(f"entire dataset: {len(self.entire_dataset)}")
        print(f"entire avg. accuracy: {self.total_acc_tracker.avg_acc1:.1f}%")

    def __run(self,
              start_index: int,
              start_time: float,
              train_sr: int,
              label_sr: int) -> Tuple[bool, int, float, float, int, int]: 
        is_done = False

        train_phase_acc_tracker = ClassificationAccuracyTracker()
        label_phase_acc_tracker = ClassificationAccuracyTracker()
        exceptional_label_phase_acc_tracker = ClassificationAccuracyTracker()
        total_acc_tracker = ClassificationAccuracyTracker()

        # train until validation is increased
        # this condition would pass train at the first train/labeling pair
        # due to no train data
        initial_state_dict = copy.deepcopy(self.student_model.module.state_dict())

        valid_metric = None
        drift_occur = False

        if self.episodic_memory.count() > 0:
            # >>>>>
            # train
            # >>>>>
            num_data_for_train = train_sr
            train_dataset, valid_dataset = self.episodic_memory.generate_dataset(config=self.config,
                                                                                 phase_index=self.phase_index,
                                                                                 entire_dataset=self.entire_dataset,
                                                                                 num_data_to_sample=num_data_for_train)
            print(f"TRAIN LENGTH: {len(train_dataset)}")
            print(f"VALID LENGTH: {len(valid_dataset)}")

            train_I_f, train_T_f, required_train_time = self.profiler.get_best_train_config(train_dataset=train_dataset)
            valid_I_f, valid_V_f, required_valid_time = self.profiler.get_best_valid_config(valid_dataset=valid_dataset)

            # >>>>> get required time for the phase and check phase budget >>>>>
            train_phase_index_slice, \
            train_start_idx, \
            train_start_time, \
            train_time_budget = split_dataset_indices_by_time(frame_latency_table=self.latency_table,
                                                              start_index=start_index,
                                                              start_time=start_time,
                                                              duration_time=required_train_time + required_valid_time)
            # <<<<< get required time for the phase and check phase budget <<<<<

            # set weights before training
            self.student_model.module.load_state_dict(initial_state_dict)

            self.student_model.infer(phase=1,
                                        dataset=self.entire_dataset,
                                        index_slice=train_phase_index_slice,
                                        phase_accuracy_tracker=train_phase_acc_tracker,
                                        phase_loss_tracker=None,
                                        window_accuracy_tracker=total_acc_tracker)
            print(f"# of processed images: {len(total_acc_tracker.corrects)}")

            if train_time_budget < (required_train_time + required_valid_time):
                # this is the end, return
                print(f"experiment ended at train phase, "
                      f"total # of images: {len(total_acc_tracker.corrects)},"
                      f"start index: {train_start_idx}")
                self.total_acc_tracker.update(acc1=total_acc_tracker.avg_acc1,
                                                acc5=0.,
                                                batch_size=total_acc_tracker.total_cnt)
                self.corrects_per_phase.append(total_acc_tracker.corrects)

                self.__write_phase_log_row(index=self.phase_index,
                                           name="last infer only at train phase",
                                           num_imgs=len(train_phase_acc_tracker.corrects),
                                           time=train_time_budget,
                                           accuracy=train_phase_acc_tracker.avg_acc1,
                                           num_train=-1,
                                           num_valid=-1,
                                           num_label=-1,
                                           buffer_count=self.episodic_memory.count(),
                                           buffer_capacity=self.episodic_memory.capacity)

                return True, -1, -1, 0, 0, 0
            else:
                valid_metric = self.student_model.train(train_dataset=train_dataset,
                                                        valid_dataset=valid_dataset,
                                                        epochs=1)
                
                self.__write_phase_log_row(index=self.phase_index,
                                           name=f"train phase",
                                           num_imgs=len(train_phase_acc_tracker.corrects),
                                           time=train_time_budget,
                                           accuracy=train_phase_acc_tracker.avg_acc1,
                                           num_train=len(train_dataset),
                                           num_valid=len(valid_dataset),
                                           num_label=-1,
                                           buffer_count=self.episodic_memory.count(),
                                           buffer_capacity=self.episodic_memory.capacity)

                print(f"train phase {required_train_time:.1f} seconds, "
                      f"# of images: {train_phase_index_slice[1] - train_phase_index_slice[0] + 1}")
                print(f"valid metric: {valid_metric:.3f}%")

                # TODO: should be parameter
                # sub_label_sr = int(432 / 2)
                sub_label_sr = self.sub_label_sr

                # >>>>> get required time for the phase and check phase budget >>>>>
                label_I_f, label_V_f, required_label_time = \
                    self.profiler.get_best_label_config(num_data_to_sample=sub_label_sr)
                
                # inference at label phase
                label_phase_index_slice, \
                next_start_idx, \
                next_start_time, \
                label_time_budget = split_dataset_indices_by_time(frame_latency_table=self.latency_table,
                                                                  start_index=train_start_idx,
                                                                  start_time=train_start_time,
                                                                  duration_time=required_label_time)

                # <<<<< get required time for the phase and check phase budget <<<<<
                self.student_model.infer(phase=2,
                                            dataset=self.entire_dataset,
                                            index_slice=label_phase_index_slice,
                                            phase_accuracy_tracker=label_phase_acc_tracker,
                                            phase_loss_tracker=None,
                                            window_accuracy_tracker=total_acc_tracker)
            

                sub_label_index_start = train_phase_index_slice[0]
                sub_label_index_end = label_phase_index_slice[1]
                index_slice_for_sampling = list(range(sub_label_index_start,
                                                        sub_label_index_end + 1))

                corrects_of_data_to_sample = []
                corrects_of_data_to_sample.extend(train_phase_acc_tracker.corrects)
                corrects_of_data_to_sample.extend(label_phase_acc_tracker.corrects)

                confidences_of_data_to_sample = []
                confidences_of_data_to_sample.extend(train_phase_acc_tracker.confidences)
                confidences_of_data_to_sample.extend(label_phase_acc_tracker.confidences)

                assert len(index_slice_for_sampling) == len(corrects_of_data_to_sample)
                print(f"IMG LENGTH TO BE SUB LABELED: {len(corrects_of_data_to_sample)}")
                print(f"SUB SAMPLE NUM: {sub_label_sr} from {len(corrects_of_data_to_sample)}")
                
                if label_time_budget < required_label_time:
                    # this is the end, return
                    print(f"experiment ended at label phase, total # of images: {len(total_acc_tracker.corrects)} start index: {train_start_idx}")
                    self.total_acc_tracker.update(acc1=total_acc_tracker.avg_acc1,
                                                    acc5=0.,
                                                    batch_size=total_acc_tracker.total_cnt)
                    self.corrects_per_phase.append(total_acc_tracker.corrects)

                    self.__write_phase_log_row(index=self.phase_index,
                                               name="last infer only at sub label phase",
                                               num_imgs=len(label_phase_acc_tracker.corrects),
                                               time=label_time_budget,
                                               accuracy=label_phase_acc_tracker.avg_acc1,
                                               num_train=-1,
                                               num_valid=-1,
                                               num_label=-1,
                                               buffer_count=self.episodic_memory.count(),
                                               buffer_capacity=self.episodic_memory.capacity)

                    return True, -1, -1, 0, 0, 0

                self.__write_phase_log_row(index=self.phase_index,
                                           name=f"label phase",
                                           num_imgs=len(label_phase_acc_tracker.corrects),
                                           time=label_time_budget,
                                           accuracy=label_phase_acc_tracker.avg_acc1,
                                           num_train=-1,
                                           num_valid=-1,
                                           num_label=sub_label_sr,
                                           buffer_count=self.episodic_memory.count(),
                                           buffer_capacity=self.episodic_memory.capacity)
                print(f"SUB LABELING TIME: {label_time_budget}")

                indices_per_label, label_metric = self.__sample_data_to_train(entire_dataset=self.entire_dataset,
                                                                              indices=index_slice_for_sampling,
                                                                              corrects=corrects_of_data_to_sample, 
                                                                              confidences=confidences_of_data_to_sample,
                                                                              label_sr=sub_label_sr)

                if label_metric - valid_metric <= -5.0:
                    print(f"DRIFT OCCUR, reset buffer: {label_metric:.1f}% - {valid_metric:.1f}% "
                          f"({label_metric - valid_metric:.1f}%)")
                    
                    drift_occur = True
                    self.episodic_memory.reset()

                self.episodic_memory.push_data(indices_per_label=indices_per_label)

        # >>>>>>>>>>>
        # fisrt label phase or extra label due to drift
        # >>>>>>>>>>>
        if self.episodic_memory.count() == 0 or drift_occur:
            if drift_occur:
                name = "extra label"
                # num_data_to_sample = (2 * label_sr) - self.episodic_memory.count()
                num_data_to_sample = (label_sr) - self.episodic_memory.count()
                exceptional_label_start_idx = next_start_idx
                exceptional_label_start_time = next_start_time
            else:
                # first label
                name = "first label"
                num_data_to_sample = self.num_data_for_first_label
                exceptional_label_start_idx = start_index
                exceptional_label_start_time = start_time

            # >>>>> get required time for the phase and check phase budget >>>>>
            label_I_f, label_V_f, required_exceptional_label_time = \
                self.profiler.get_best_label_config(num_data_to_sample=num_data_to_sample)
            
            # inference at label phase
            exceptional_label_phase_index_slice, \
            next_start_idx, \
            next_start_time, \
            exceptional_label_time_budget = split_dataset_indices_by_time(frame_latency_table=self.latency_table,
                                                                          start_index=exceptional_label_start_idx,
                                                                          start_time=exceptional_label_start_time,
                                                                          duration_time=required_exceptional_label_time)
            # <<<<< get required time for the phase and check phase budget <<<<<
            
            self.student_model.infer(phase=2,
                                     dataset=self.entire_dataset,
                                     index_slice=exceptional_label_phase_index_slice,
                                     phase_accuracy_tracker=exceptional_label_phase_acc_tracker,
                                     phase_loss_tracker=None,
                                     window_accuracy_tracker=total_acc_tracker)
            # print(f"# of processed images: {len(total_acc_tracker.corrects)}")

            if exceptional_label_time_budget < required_exceptional_label_time:
                # this is the end, return
                print(f"experiment ended at {name}, "
                      f"total # of images: {len(total_acc_tracker.corrects)} start index: {train_start_idx}")
                self.total_acc_tracker.update(acc1=total_acc_tracker.avg_acc1,
                                              acc5=0.,
                                              batch_size=total_acc_tracker.total_cnt)
                self.corrects_per_phase.append(total_acc_tracker.corrects)

                self.__write_phase_log_row(index=self.phase_index,
                                           name=f"last infer only at {name}",
                                           num_imgs=len(exceptional_label_phase_acc_tracker.corrects),
                                           time=exceptional_label_time_budget,
                                           accuracy=exceptional_label_phase_acc_tracker.avg_acc1,
                                           num_train=-1,
                                           num_valid=-1,
                                           num_label=-1,
                                           buffer_count=self.episodic_memory.count(),
                                           buffer_capacity=self.episodic_memory.capacity)

                return True, -1, -1, 0, 0, 0

            self.__write_phase_log_row(index=self.phase_index,
                                       name=name,
                                       num_imgs=len(exceptional_label_phase_acc_tracker.corrects),
                                       time=exceptional_label_time_budget,
                                       accuracy=exceptional_label_phase_acc_tracker.avg_acc1,
                                       num_train=-1,
                                       num_valid=-1,
                                       num_label=num_data_to_sample,
                                       buffer_count=self.episodic_memory.count(),
                                       buffer_capacity=self.episodic_memory.capacity)

            # update episodic memory -> labeling
            index_slice_for_sampling = list(range(exceptional_label_phase_index_slice[0],
                                                  exceptional_label_phase_index_slice[1] + 1))
            corrects_of_data_to_sample = exceptional_label_phase_acc_tracker.corrects
            confidences_of_data_to_sample = exceptional_label_phase_acc_tracker.confidences

            indices_per_label, label_metric = self.__sample_data_to_train(entire_dataset=self.entire_dataset,
                                                                          indices=index_slice_for_sampling,
                                                                          corrects=corrects_of_data_to_sample, 
                                                                          confidences=confidences_of_data_to_sample,
                                                                          label_sr=num_data_to_sample)
            
            self.episodic_memory.push_data(indices_per_label=indices_per_label)

        next_train_sr, next_label_sr = self.profiler.schedule(prev_data_metric=valid_metric,
                                                              curr_data_metric=label_metric)

        # >>>>> log >>>>>
        if self.episodic_memory.count() > 0:
            print(f"train phase metric: {train_phase_acc_tracker.avg_acc1:.1f}%")
            print(f"label phase metric: {label_phase_acc_tracker.avg_acc1:.1f}%")

            if drift_occur:
                print(f"extra label metric:  {exceptional_label_phase_acc_tracker.avg_acc1:.1f}%")    
        else:
            print(f"first label metric:  {exceptional_label_phase_acc_tracker.avg_acc1:.1f}%")

        print(f"total phase metric:  {total_acc_tracker.avg_acc1:.1f}%")

        # if prev_data_metric is not None:
        #     print(f">>>>> SCHEDULED RESULT >>>>>")
        #     print(f"Diff. {curr_data_metric:.1f}% - {prev_data_metric:.1f}% = {curr_data_metric-prev_data_metric:.1f}%")
        #     print(f"TRAIN DATA: {train_sr} -> {next_train_sr}")
        #     print(f"LABEL DATA: {label_sr} -> {next_label_sr}")
        #     print(f"<<<<< SCHEDULED RESULT <<<<<")
        # else:
        #     print(f">>>>> SCHEDULED RESULT >>>>>")
        #     print(f"TRAIN DATA: {train_sr} -> {next_train_sr}")
        #     print(f"LABEL DATA: {label_sr} -> {next_label_sr}")
        #     print(f"<<<<< SCHEDULED RESULT <<<<<")

        self.total_acc_tracker.update(acc1=total_acc_tracker.avg_acc1,
                                      acc5=0.,
                                      batch_size=total_acc_tracker.total_cnt)
        
        print(f"# OF IMAGES THIS PHASE: {len(total_acc_tracker.corrects)}")
        self.corrects_per_phase.append(total_acc_tracker.corrects)

        total_corrects = []
        for c in self.corrects_per_phase:
            total_corrects.extend(c)

        num_window = self.config.window_time * self.config.fps
        for w in range(0, len(total_corrects), num_window):
            corrects_of_phase = total_corrects[w:w+(num_window)]

            if len(corrects_of_phase) == num_window:
                print(f"window #{int(w / num_window)}: {np.sum(corrects_of_phase) / len(corrects_of_phase) * 100:.1f}%")
        # <<<<< log <<<<<

        if next_start_idx == -1:
            is_done = True
        
        return is_done, next_start_idx, next_start_time, 0, next_train_sr, next_label_sr

    def __write_phase_log_row(self,
                              index: int,
                              name: str,
                              num_imgs: int,
                              time: float,
                              accuracy: float,
                              num_train: int,
                              num_valid: int,
                              num_label: int,
                              buffer_count: int,
                              buffer_capacity: int):
        with open(self.phase_log_path, "a") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow([
                index,
                name,
                num_imgs,
                time, 
                accuracy,
                num_train,
                num_valid,
                num_label,
                buffer_count,
                buffer_capacity
            ])

    def __sample_data_to_train(self,
                               entire_dataset: MystiqueDataset,
                               indices: List[int],
                               corrects: List[int],
                               confidences: List[float],
                               label_sr: int) -> Tuple[List[List[int]], float]:
        if self.config.use_active_cl:
            return self.__sample_data_to_train_active_cl(entire_dataset=entire_dataset,
                                                         indices=indices,
                                                         corrects=corrects,
                                                         confidences=confidences,
                                                         label_sr=label_sr)
        else:
            return self.__sample_data_to_train_uniform(entire_dataset=entire_dataset,
                                                       indices=indices,
                                                       corrects=corrects,
                                                       label_sr=label_sr)

    def __sample_data_to_train_uniform(self,
                                       entire_dataset: MystiqueDataset,
                                       indices: List[int],
                                       corrects: List[int],
                                       label_sr: int) -> Tuple[List[List[int]], float]:
        assert (len(indices) == len(corrects))

        total_num = len(indices)
        num_imgs_to_sample = label_sr # int(math.floor(total_num * label_sr / 100.))
        step = int(np.floor(total_num / num_imgs_to_sample))

        print(f"[SAMPLING] # of images: {total_num}, trying to sample: {num_imgs_to_sample} images...")

        sampled_indices = []
        sampled_corrects = []

        if step >= 1:
            for offset in range(0, indices[-1] - indices[0] + 1, step):
                if len(sampled_indices) == num_imgs_to_sample:
                    break
                
                sampled_indices.append(indices[offset])
                sampled_corrects.append(corrects[offset])
        else:
            sampled_indices = list(range(total_num))

        print(f"# of images: {total_num}, sampled # of images: {len(sampled_indices)} ({len(sampled_indices) / total_num * 100.:.1f}%)")

        train_dataset = TrainSampleDataset(ori_dataset=entire_dataset,
                                           indices=sampled_indices)
        
        indices_per_label = [[] for _ in range(self.config.num_classes)]

        assert (len(train_dataset) == len(sampled_indices))

        train_sample_cnt = 0
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
                indices_per_label[label].append(sampled_indices[train_sample_cnt])
                train_sample_cnt += 1

        assert (len(train_dataset) == int(np.sum([len(v) for v in indices_per_label])))

        return indices_per_label, np.sum(sampled_corrects) / len(sampled_corrects) * 100.


    def __sample_data_to_train_active_cl(self,
                                         entire_dataset: MystiqueDataset,
                                         indices: List[int],
                                         corrects: List[int],
                                         confidences: List[float],
                                         label_sr: int) -> Tuple[List[List[int]], float]:
        assert (len(indices) == len(corrects))
        assert (len(indices) == len(confidences))

        candidate_samples = []
        for i in range(len(indices)):
            candidate_samples.append({
                "index": indices[i],
                "correct": corrects[i],
                "confidence": confidences[i]
            })
        
        confidence_pivot = 0.5
        candidate_samples = sorted(candidate_samples,
                                   key=lambda x: abs(x["confidence"] - confidence_pivot))

        total_num = len(indices)
        num_imgs_to_sample = label_sr # int(math.floor(total_num * label_sr / 100.))

        print(f"[SAMPLING] # of images: {total_num}, trying to sample: {num_imgs_to_sample} images...")

        sampled_indices = []
        sampled_corrects = []

        candidate_samples = candidate_samples[0:num_imgs_to_sample]

        for i in range(len(candidate_samples)):
            sampled_indices.append(candidate_samples[i]["index"])
            sampled_corrects.append(candidate_samples[i]["correct"])

        print(f"# of images: {total_num}, sampled # of images: {len(sampled_indices)} ({len(sampled_indices) / total_num * 100.:.1f}%)")

        train_dataset = TrainSampleDataset(ori_dataset=entire_dataset,
                                           indices=sampled_indices)
        
        indices_per_label = [[] for _ in range(self.config.num_classes)]

        assert (len(train_dataset) == len(sampled_indices))

        train_sample_cnt = 0
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
                indices_per_label[label].append(sampled_indices[train_sample_cnt])
                train_sample_cnt += 1

        assert (len(train_dataset) == int(np.sum([len(v) for v in indices_per_label])))

        return indices_per_label, np.sum(sampled_corrects) / len(sampled_corrects) * 100.


if __name__ == "__main__":
    args = parser.parse_args()
    emulator = Emulator(config=Config(config_file_path=args.config_path))
    
    torch.cuda.synchronize()
    start_time = time.time()

    emulator.run()
    
    torch.cuda.synchronize()
    end_time = time.time()

    print(f"experiment time: {end_time-start_time:.1f} seconds")