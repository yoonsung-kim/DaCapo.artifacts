import csv
import math
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from torch.utils.data import DataLoader
from emulator.config import Config
from emulator.dataset import TrainSampleDataset, MystiqueDataset


class EpisodicMemory:
    def __init__(self, num_classes, capacity):
        self.num_classes = num_classes
        self.capacity = capacity
        self.num_valid_imgs = 108

        self.indices: List[int] = []
        self.labels: List[int] = []

    def reset(self):
        self.indices: List[int] = []
        self.labels: List[int] = []

    def push_data(self, indices_per_label: List[List[int]]):
        # if len(indices_per_label) != len(self.indices):
        #     raise ValueError(f"length mismatched -> "
        #                      f"indices_per_label: {len(indices_per_label)}, "
        #                      f"self.indices: {len(self.indices)}")
        
        len_per_label = [len(indices_per_label[l]) for l in range(len(indices_per_label))]
        cnt_per_label = [0 for _ in range(len(indices_per_label))]

        cnt = 0
        while True:
            is_done = True

            for l in range(self.num_classes):
                if cnt_per_label[l] < len_per_label[l]:
                    is_done = False

                    self.indices.insert(0, indices_per_label[l][cnt_per_label[l]])
                    self.labels.insert(0, l)
                    cnt_per_label[l] += 1
                    cnt += 1

            if is_done:
                break

        assert (len(self.indices) == len(self.labels))

        if len(self.indices) > self.capacity:
            self.indices = self.indices[0:self.capacity]
            self.labels = self.labels[0:self.capacity]

        print(f"UPDATED EM: {len(self.indices)} / {self.capacity}")

    def count(self):
        assert (len(self.indices) == len(self.labels))

        return len(self.indices)

    def generate_dataset(self,
                         config: Config,
                         phase_index: int,
                         entire_dataset: MystiqueDataset,
                         num_data_to_sample: int) -> Tuple[TrainSampleDataset]:
        indices_per_label = [[] for _ in range(self.num_classes)]
        for i in range(len(self.labels)):
            index = self.indices[i]
            label = self.labels[i]

            indices_per_label[label].append(index)
        
        # >>>>> TODO: step sampling
        ratio_per_label = [0. for _ in range(self.num_classes)]

        for l in range(self.num_classes):
            ratio_per_label[l] = len(indices_per_label[l]) / len(self.indices)

        num_to_sample = [int(math.floor(num_data_to_sample * ratio_per_label[l])) for l in range(self.num_classes)]

        for l in range(self.num_classes):
            print(f"LABEL #{l}: {num_to_sample[l]} ({ratio_per_label[l] * 100:.1f}%), {len(indices_per_label[l])}")

        print(f"EM size: {len(self.indices)}, num_data_to_sample: {num_data_to_sample}")
        while int(np.sum(num_to_sample)) != num_data_to_sample:
            for l in range(self.num_classes):
                if int(np.sum(num_to_sample)) == num_data_to_sample:
                    break

                if num_to_sample[l] + 1 <= len(indices_per_label[l]):
                    num_to_sample[l] += 1
        # <<<<<

        indices = []
        labels = []

        for l in range(self.num_classes):
            if num_to_sample[l] == 0:
                continue

            step = int(math.floor(len(indices_per_label[l]) / num_to_sample[l]))
            
            cnt = 0
            print(f"label: {l} ({len(indices_per_label[l])}, {num_to_sample[l]})")
            for offset in range(0, len(indices_per_label[l]), step):
                if num_to_sample[l] == cnt:
                    # print(f"label: #{l}: {cnt}")
                    break
                # else:
                #     print(f"label: #{l}: {cnt}")


                indices.append(indices_per_label[l][offset])          
                labels.append(l)
        
                cnt += 1

        assert (len(labels) == len(indices))
        print(f"{np.sum(num_to_sample)}")
        print(f"labels: {len(labels)}, num data to sample: {num_data_to_sample}")
        assert (len(labels) == num_data_to_sample)

        cnt_per_label = [0 for _ in range(self.num_classes)]
        # RR: pre-knowledge for model
        while len(indices) != num_data_to_sample:
            for l in range(len(indices_per_label)):
                if len(indices) == num_data_to_sample:
                    break

                if cnt_per_label[l] < len(indices_per_label[l]):
                    indices.append(indices_per_label[l][cnt_per_label[l]])          
                    labels.append(l)
                    cnt_per_label[l] += 1
        
        assert (len(labels) == len(indices))
        indices_per_label = [[] for _ in range(self.num_classes)]
        for i in range(len(labels)):
            indices_per_label[labels[i]].append(indices[i])

        print(f"total sampled # images: {len(indices)}")

        # >>> TODO: remove >>>
        # print(f"CURRENT EM")
        # for l in range(len(indices_per_label)):
        #     print(f"SAMPLED DATA LABEL #{l}: {len(indices_per_label[l])}")
        # <<< TODO: remove <<< 

        indices_per_label = [[] for _ in range(self.num_classes)]
        for i in range(len(labels)):
            index = indices[i]
            label = labels[i]

            indices_per_label[label].append(index)

        train_indices = []
        valid_indices = []

        ratio = self.num_valid_imgs / num_data_to_sample
        # ratio = self.num_valid_imgs / len(self.indices) # num_data_to_sample

        for l in range(self.num_classes):
            num_valid = int(math.floor(len(indices_per_label[l]) * ratio))
            step = int(math.floor(1 / ratio))
            
            indices = []

            for offset in range(0, len(indices_per_label[l]), step):
                indices.append(indices_per_label[l][offset])

                if len(indices) == num_valid:
                    break

            valid_indices.extend(indices)
            train_indices.extend([item for item in indices_per_label[l] if item not in indices])

        train_dataset = TrainSampleDataset(ori_dataset=entire_dataset,
                                           indices=train_indices)
        valid_dataset = TrainSampleDataset(ori_dataset=entire_dataset,
                                           indices=valid_indices)
        
        print(f">>>>> CURRENT DS FROM BUFFER >>>>>")
        print(f"SELECTED: {len(train_indices) + len(valid_indices)} BUFFER COUNT: {len(self.indices)}, CAPACITY: {self.capacity}")
        print(f"TRAIN: {len(train_indices)}")
        print(f"VALID: {len(valid_indices)}")
        print(f"<<<<< CURRENT DS FROM BUFFER <<<<<")
        
        # >>>>> log >>>>>
        sample_cnt_per_label = {l: 0 for l in range(self.num_classes)}

        data_loader = DataLoader(dataset=train_dataset,
                                 shuffle=False,
                                 pin_memory=True,
                                 batch_size=config.infer_batch_size,
                                 num_workers=config.num_workers)
        
        for _, targets, _ in tqdm(data_loader,
                                  desc="Checking dist. of sampled train data",
                                  unit=" batches"):
            for b in range(targets.shape[0]):
                label = targets[b].item()
                sample_cnt_per_label[label] += 1

        sampled_num = len(train_dataset)
        print(f"sampled_num: {sampled_num}")
        row = [
            phase_index,
            sampled_num
        ]
        
        print(f"# of train data: {sampled_num} images")
        for l in sample_cnt_per_label.keys():
            # print(f"label #{l}: {sample_cnt_per_label[l]} images ({sample_cnt_per_label[l] / sampled_num * 100.:.1f}%)", end=" ")
            print(f"label #{l}: {sample_cnt_per_label[l]} images", end=" | ")
            row.append(sample_cnt_per_label[l])
            row.append(sample_cnt_per_label[l] / sampled_num * 100.)
        print()

        with open(f"{config.output_root}/train_sample_dist_log.csv", "a") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(row)
        # <<<<< log <<<<<

        # >>>>> log >>>>>
        sample_cnt_per_label = {l: 0 for l in range(self.num_classes)}

        data_loader = DataLoader(dataset=valid_dataset,
                                 shuffle=False,
                                 pin_memory=True,
                                 batch_size=config.infer_batch_size,
                                 num_workers=config.num_workers)
        
        for _, targets, _ in tqdm(data_loader,
                                  desc="Checking dist. of sampled valid data",
                                  unit=" batches"):
            for b in range(targets.shape[0]):
                label = targets[b].item()
                sample_cnt_per_label[label] += 1

        sampled_num = len(valid_dataset)
        row = [
            phase_index,
            sampled_num
        ]
        
        print(f"# of valid data: {sampled_num} images")
        for l in sample_cnt_per_label.keys():
            # print(f"label #{l}: {sample_cnt_per_label[l]} images ({sample_cnt_per_label[l] / sampled_num * 100.:.1f}%)", end=" ")
            print(f"label #{l}: {sample_cnt_per_label[l]}", end=" | ")
            row.append(sample_cnt_per_label[l])
            row.append(sample_cnt_per_label[l] / sampled_num * 100.)
        print()
        
        with open(f"{config.output_root}/valid_sample_dist_log.csv", "a") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(row)
        # <<<<< log <<<<<
        
        return train_dataset, valid_dataset