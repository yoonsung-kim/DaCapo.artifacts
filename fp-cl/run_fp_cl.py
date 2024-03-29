import os
import csv
import sys
import copy
import json
import math
import torch
import shutil
import timm
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from pathlib import Path
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from util.reproduce import set_reproducibility, seed_worker
from util.accuracy import calculate_accuracy, ClassificationAccuracyTracker
from fp_cl.revision.dataset_util import generate_windows
from fp_cl.revision.model import ModelGenertor
from scipy.stats import gmean
from torchsampler import ImbalancedDatasetSampler
from fp_cl.submission.dataset_util import split_dataset_indices_by_time, CustomSampler, MystiqueDataset, TrainSampleDataset

FPS = 30
WINDOW_TIME = 120
NUM_CLASSES = 9
NUM_WORKERS = 16
TRAIN_BATCH_SIZE = 16
INFER_BATCH_SIZE = 64


parser = argparse.ArgumentParser(description="Continual learning to get optimal accuracy")
parser.add_argument("--scenario-path", type=str, default=None, help="scenario directory")
parser.add_argument("--student", type=str, default=None, help="student model")
parser.add_argument("--weight-path", type=str, default=None, help="weight path of student model")
parser.add_argument("--output-root", type=str, default=None, help="root directory of output")
parser.add_argument("--max-epoch", type=int, default=None, help="maximum epochs")
parser.add_argument("--seed", type=int, default=128, help="seed")


def run_revision_version(scenario_path,
                         model_name,
                         weight_path,
                         output_root,
                         max_epoch):
    # >>> generate windows >>>
    windows = generate_windows(scenario_path=Path(scenario_path),
                               fps=FPS,
                               window_time=WINDOW_TIME)
    num_windows = len(windows)
    print(f"# of windows: {num_windows}")

    # >>> generate model >>>
    model = ModelGenertor.generate(name=model_name,
                                   num_classes=NUM_CLASSES,
                                   weight_path=weight_path).to(device=device)
    # <<< generate model <<<

    output_file_path = output_root / "result.csv"

    with open(output_file_path, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["window index", "accuracy", "epoch"])

    infer_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    for window_idx in range(num_windows):
        infer_dataset = windows[window_idx]
        infer_dataset.transform = infer_transform
        infer_loader = DataLoader(dataset=infer_dataset,
                                  batch_size=INFER_BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True,
                                  drop_last=False)
        
        best_accuracy = -1
        best_epoch = -1
        best_state_dict = copy.deepcopy(model.cpu().state_dict())

        if window_idx > 0:
            train_dataset = windows[window_idx-1]

            train_dataset.transform = train_transform

            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=TRAIN_BATCH_SIZE,
                                      sampler=ImbalancedDatasetSampler(train_dataset),
                                      # shuffle=True,
                                      num_workers=NUM_WORKERS,
                                      pin_memory=True,
                                      drop_last=False)
            
            criterion = torch.nn.CrossEntropyLoss().to(device=device)
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=0.007,
                                        momentum=0.9,
                                        weight_decay=1e-4)

            for epoch in tqdm(range(max_epoch),
                              desc=f"retraining at window #{window_idx+1}/{num_windows}",
                              unit=" epochs"):
                model = model.to(device)

                # retrain
                model.train()
                for inputs, targets in tqdm(train_loader,
                                            desc=f"retrain of epoch #{epoch+1}/{max_epoch} "
                                                 f"at window #{window_idx+1}/{num_windows}",
                                            unit=" batches",
                                            leave=False):
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs)

                    loss = criterion(outputs, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # infer
                acc_tracker = ClassificationAccuracyTracker()

                model.eval()
                with torch.no_grad():
                    for inputs, targets in tqdm(infer_loader,
                                                desc=f"valid of epoch #{epoch+1}/{max_epoch} "
                                                     f"at window #{window_idx+1}/{num_windows}",
                                                unit=" batches",
                                                leave=False):
                        inputs, targets = inputs.to(device), targets.to(device)

                        outputs = model(inputs)

                        top1_acc, _, _ = calculate_accuracy(outputs, targets, topk_list=[1])
                        top1_acc = top1_acc[0]
                
                        acc_tracker.update(top1_acc, 0, inputs.shape[0])
                
                current_acc = acc_tracker.avg_acc1
                if current_acc > best_accuracy:
                    best_accuracy = current_acc
                    best_epoch = epoch + 1
                    best_state_dict = copy.deepcopy(model.cpu().state_dict())

            model.load_state_dict(best_state_dict)
    
            print(f"window #{window_idx} best accuracy: {best_accuracy:.1f}% at epoch {best_epoch}")
        else:
            print(f"the fisrt window, nothing to report")

        with open(output_file_path, "a") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([window_idx, best_accuracy, best_epoch])


def sample_data_to_train(entire_dataset: MystiqueDataset,
                         index_slice: Tuple[int, int]) -> Dataset:
    total_num = index_slice[1] - index_slice[0] + 1
    train_sample_indices = list(range(0, total_num))

    train_dataset = TrainSampleDataset(ori_dataset=entire_dataset,
                                        indices=train_sample_indices)
    
    sample_cnt_per_label = {i: 0 for i in range(NUM_CLASSES)}

    data_loader = DataLoader(dataset=entire_dataset,
                             sampler=CustomSampler(indices=list(range(index_slice[0], index_slice[1] + 1))),
                             shuffle=False,
                             pin_memory=True,
                             batch_size=INFER_BATCH_SIZE,
                             num_workers=NUM_WORKERS)

    for inputs, targets, _ in tqdm(data_loader,
                                   desc="Checking sampled train dataset",
                                   unit=" batches"):
        for b in range(inputs.shape[0]):
            label = targets[b].item()
            sample_cnt_per_label[label] += 1

    return train_dataset


def run_submission_version(scenario_path,
                           model_name,
                           weight_path,
                           output_root,
                           max_epoch):
    # >>> generate windows >>>
    entire_dataset = MystiqueDataset(scenario_dir=Path(scenario_path),
                                     transform=None)
    latency_table = entire_dataset.generate_frame_latency_table(fps=FPS)

    start_idx = 0
    start_time = 0
    duration_time = WINDOW_TIME

    window_index_slices: List[Tuple[int, int]] = []
    train_dataset_from_previous_window: List[TrainSampleDataset] = [None,]

    while start_idx != -1:
        index_slice, \
        next_start_idx, \
        next_start_time, \
        time = split_dataset_indices_by_time(frame_latency_table=latency_table,
                                             start_index=start_idx,
                                             start_time=start_time,
                                             duration_time=duration_time)

        start_idx = next_start_idx
        start_time = next_start_time
        
        window_index_slices.append(index_slice)
    # <<< generate windows <<<

    # >>> generate model >>>
    model = ModelGenertor.generate(name=model_name,
                                   num_classes=NUM_CLASSES,
                                   weight_path=weight_path).to(device=device)
    model = model.to(device=device)
    # <<< generate model <<<

    output_file_path = output_root / "result.csv"

    with open(output_file_path, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["window index", "accuracy", "epoch"])

    # >>> train & save weights >>>
    num_windows = len(window_index_slices)
    for w in range(num_windows):
        train_dataset = train_dataset_from_previous_window[w]
        infer_index_slices = window_index_slices[w]

        best_accuracy = -1
        best_epoch = -1
        best_state_dict = copy.deepcopy(model.cpu().state_dict())

        if train_dataset is not None:
            # train
            g = torch.Generator()
            g.manual_seed(seed)
            
            criterion = torch.nn.CrossEntropyLoss().to(device=device)

            lrs = {
                "resnet18": 0.007,
                "resnet34": 0.01,
            }

            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=lrs[model_name],
                                        momentum=0.9,
                                        weight_decay=1e-4)

            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

            train_dataset.transform = train_transform

            train_data_loader = DataLoader(dataset=train_dataset,
                                           num_workers=NUM_WORKERS,
                                           sampler=ImbalancedDatasetSampler(train_dataset),
                                           # shuffle=False,
                                           pin_memory=True,
                                           batch_size=TRAIN_BATCH_SIZE,
                                           worker_init_fn=seed_worker,
                                           generator=g)
            
            entire_dataset.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            valid_data_loader = DataLoader(dataset=entire_dataset,
                                           sampler=CustomSampler(indices=list(range(infer_index_slices[0],
                                                                                    infer_index_slices[1] + 1))),
                                           num_workers=NUM_WORKERS,
                                           pin_memory=True,
                                           shuffle=False,
                                           batch_size=INFER_BATCH_SIZE,
                                           worker_init_fn=seed_worker,
                                           generator=g)
            
            for e in tqdm(range(max_epoch),
                          desc=f"retraining at window #{w+1}/{num_windows}",
                          unit=" epochs"):
                model = model.to(device)
                model.train()
                for inputs, targets, _ in tqdm(train_data_loader,
                                            desc=f"retrain of epoch #{e+1}/{max_epoch} "
                                                 f"at window #{w+1}/{num_windows}",
                                            unit=" batches",
                                            leave=False):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                acc_tracker = ClassificationAccuracyTracker()

                model.eval()
                for inputs, targets, _ in tqdm(valid_data_loader,
                                               desc=f"window #{w} validation with current window",
                                               leave=False):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                                
                    accs, _, _ = calculate_accuracy(outputs, targets, topk_list=[1,])
                    acc_tracker.update(accs[0], 0., batch_size=inputs.shape[0])

                if acc_tracker.avg_acc1 > best_accuracy:
                    best_accuracy = acc_tracker.avg_acc1
                    best_epoch = e + 1
                    best_state_dict = copy.deepcopy(model.cpu().state_dict())

        model.load_state_dict(best_state_dict)
        print(f"window #{w} best accuracy: {best_accuracy:.1f}% at epoch {best_epoch}")

        # log ratio of sampled rate
        train_dataset = sample_data_to_train(entire_dataset=entire_dataset,
                                             index_slice=infer_index_slices)
        train_dataset_from_previous_window.append(train_dataset)

        with open(output_file_path, "a") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([w, best_accuracy, best_epoch])


if __name__ == "__main__":
    args = parser.parse_args()
    scenario_path = args.scenario_path
    model_name = args.student
    weight_path = args.weight_path
    output_root = Path(args.output_root)
    max_epoch = args.max_epoch
    seed = args.seed

    set_reproducibility(seed)

    device = torch.device("cuda:0")

    if "vit" in model_name:
        run_revision_version(scenario_path,
                             model_name,
                             weight_path,
                             output_root,
                             max_epoch)
    else:
        run_submission_version(scenario_path,
                               model_name,
                               weight_path,
                               output_root,
                               max_epoch)