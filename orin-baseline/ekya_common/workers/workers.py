import ray
import time
import math
import copy
import torch
import random
import numpy as np
import torch.nn as nn
from typing import List
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from src.util.accuracy import calculate_accuracy, ClassificationAccuracyTracker
from torchvision.models import resnet18, resnet101 #TODO: make model generator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.util.reprod_util import set_reproducibility
# from dummy.smart_sample import get_smart_indices
# from util.bfp_model_converter import BfpModelConverter

NUM_GPUS_PER_JOB = 1e-4

@ray.remote(num_gpus=NUM_GPUS_PER_JOB)
class ModelContainer:
    def __init__(self, model: nn.Module):
        self.is_model_updated = False
        self.model_state_dict = model.state_dict()
        self.fp16_available = torch.cuda.is_available() and hasattr(torch.cuda, "amp")

    def check_model_update(self):
        return self.is_model_updated

    def pop_model_state_dict(self):
        self.is_model_updated = False
        return self.model_state_dict

    def push_model_state_dict(self, model_state_dict):
        self.model_state_dict = model_state_dict
        self.is_model_updated = True

    def get_model_state_dict(self):
        return self.model_state_dict

    def set_model_state_dict(self, model_state_dict):
        self.model_state_dict = model_state_dict
        self.is_model_updated = False

def set_mps(gpu_allocation):
    import os
    
    gpu_allocation = min(100, gpu_allocation)
    assert 0 < gpu_allocation, f"Invalid GPU allocation: {gpu_allocation}"
    print(f"DEBUG: Setting CUDA MPS alloc to {gpu_allocation} (before: {os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', 100)})")
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(gpu_allocation)
    
@ray.remote(num_gpus=NUM_GPUS_PER_JOB)
class TrainWorker:
    def __init__(self, config, model, model_container, gpu_allocation=100):
        set_mps(gpu_allocation)
        
        is_fp16_available = torch.cuda.is_available() and hasattr(torch.cuda, "amp")
        use_fp16 = True if is_fp16_available else False
        
        self.seed = config["seed"]
        self.model = copy.deepcopy(model).cuda()
        
        if config["use_bfp"]:
            model_converter = BfpModelConverter()
            model_converter.convert(module=self.model, ratio=1.0)

        self.model = self.model.cuda()

        self.epochs = config["train_epochs"]
        self.batch_size = config["train_batch_size"]
        self.window_time = config["window_time"]
        self.model_container = model_container
        self.use_fp16 = use_fp16
        self.gradient_scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
        self.update_weight_within_window = True
        self.lr = config["lr"]
        self.model.load_state_dict(ray.get(self.model_container.get_model_state_dict.remote()))

    def train(self, train_dataset: TensorDataset, start_time: int):
        set_reproducibility(self.seed)

        if train_dataset is None:
            results = {
                "iterations_per_epoch": 0,
                "total_iterations": 0,
                "processed_iterations": 0,
                "processed_iterations_ratio": 0,
                "retrained_time": 0,
            }

            return results

        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.lr,
                                    momentum=0.9,
                                    weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, "min")

        self.model.train()

        data_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
        )

        iterations_per_epoch = len(data_loader)
        total_iterations = self.epochs * iterations_per_epoch

        iter_cnt = 0
        should_stop = False
        for _ in range(self.epochs):
            losses = []

            for inputs, targets in data_loader:
                inputs, targets = inputs.to("cuda", non_blocking=True), targets.to("cuda", non_blocking=True)

                with autocast(enabled=self.use_fp16):
                    outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                self.gradient_scaler.scale(loss).backward()
                self.gradient_scaler.step(optimizer)
                self.gradient_scaler.update()
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                iter_cnt += 1
                losses.append(loss.item())
                
                iter_time = time.time_ns()
                exec_time = (iter_time - start_time) / 1000_000_000

                # print(f"train -> {iter_cnt}/{total_iterations}")

                if exec_time >= self.window_time:
                    end_time = time.time_ns()
                    should_stop = True
                    break

            # scheduler.step(torch.mean(torch.tensor(losses)))

            if should_stop:
                break
        
        if not should_stop:
            end_time = time.time_ns()

        if self.update_weight_within_window:
            ray.get(self.model_container.push_model_state_dict.remote(self.model.state_dict()))

        results = {
            "iterations_per_epoch": iterations_per_epoch,
            "total_iterations": total_iterations,
            "processed_iterations": iter_cnt,
            "processed_iterations_ratio": iter_cnt / total_iterations,
            "retrained_time": (end_time - start_time) / 1000_000_000,
        }

        return results
    
    def get_model_state_dict(self):
        return self.model.state_dict()


@ray.remote(num_gpus=NUM_GPUS_PER_JOB)
class InferenceWorker:
    def __init__(self, config, model, model_container, gpu_allocation=100):
        set_mps(gpu_allocation)
        
        is_fp16_available = torch.cuda.is_available() and hasattr(torch.cuda, "amp")
        use_fp16 = True if is_fp16_available else False        
        
        self.seed = config["seed"]
        self.model = copy.deepcopy(model).cuda()

        if config["use_bfp"]:
            model_converter = BfpModelConverter()
            model_converter.convert(module=self.model, ratio=1.0)

        self.model = self.model.cuda()

        self.batch_size = config["inference_batch_size"]
        self.window_time = config["window_time"]
        self.model_container = model_container
        self.use_fp16 = use_fp16
        self.update_weight_within_window = True
        self.model.load_state_dict(ray.get(self.model_container.get_model_state_dict.remote()))

    def inference(self, inference_dataset: Subset, start_time: int):
        set_reproducibility(self.seed)

        data_loader = DataLoader(
            inference_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=16,  
            pin_memory=True
        )

        acc_tracker = ClassificationAccuracyTracker()
        acc_tracker.reset()

        criterion = nn.CrossEntropyLoss().cuda()

        iter_cnt = 0
        self.model.eval()
        total_iterations = len(data_loader)
        
        acc_nums = []
        acc_times = []

        avg_acc_per_label = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
        }

        data_cnt_per_label = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
        }

        with torch.no_grad():
            for inputs, targets in data_loader:
                # update model if its training is done
                if self.update_weight_within_window:
                    if ray.get(self.model_container.check_model_update.remote()):
                        params = ray.get(self.model_container.pop_model_state_dict.remote())
                        self.model.load_state_dict(params)
                        self.model.eval()
                        print(f"model updated")

                inputs, targets = inputs.to("cuda", non_blocking=True), targets.to("cuda", non_blocking=True)

                with autocast(enabled=self.use_fp16):
                    outputs = self.model(inputs)

                loss = criterion(outputs, targets)

                acc = calculate_accuracy(outputs, targets, (1,))
                acc1 = acc[0]
                acc_tracker.update(acc1=acc1, acc5=0, batch_size=self.batch_size)
                
                target_label = targets.item()

                # statistics per label
                avg_acc_per_label[target_label].append(acc1)
                data_cnt_per_label[target_label] += 1

                iter_cnt += 1
                iter_time = time.time_ns()
                exec_time = (iter_time - start_time) / 1000_000_000

                acc_nums.append(acc1)
                acc_times.append(iter_time)

                # print(f"inference -> {iter_cnt}/{total_iterations} (acc1: {acc1:.2f}%)")
                if exec_time >= self.window_time:
                    break

        drop_ratio = (1. - (iter_cnt / total_iterations)) * 100.
        avg_acc1 = acc_tracker.avg_acc1
        for _ in range(total_iterations - iter_cnt):
            acc_tracker.update(acc1=0, acc5=0, batch_size=self.batch_size)
        total_avg_acc1 = acc_tracker.avg_acc1

        results = {
            "total_iterations": total_iterations,
            "processed_iterations": iter_cnt,
            "processed_iterations_ratio": iter_cnt / total_iterations,
            "drop_ratio": drop_ratio,
            "avg_acc1": avg_acc1,
            "total_avg_acc1": total_avg_acc1,
            "acc_nums": acc_nums,
            "acc_times": acc_times,
            "data_cnt_per_label": data_cnt_per_label
        }

        return results
    
    def set_model_state_dict(self, model_state_dict):
        self.model.load_state_dict(model_state_dict)


@ray.remote(num_gpus=NUM_GPUS_PER_JOB)
class TeacherWorker:
    def __init__(self, config, model, gpu_allocation=100):
        set_mps(gpu_allocation)
        
        is_fp16_available = torch.cuda.is_available() and hasattr(torch.cuda, "amp")
        use_fp16 = True if is_fp16_available else False
        
        self.seed = config["seed"]
        self.model = model.cuda()
        
        if config["use_bfp"]:
            model_converter = BfpModelConverter()
            model_converter.convert(module=self.model, ratio=1.0)

        self.batch_size = config["teacher_batch_size"]
        self.window_time = config["window_time"]
        self.num_imgs_per_window = config["num_imgs_per_window"]
        self.num_train_data = int(self.num_imgs_per_window * config["train_data_sampling_rate"])
        self.use_fp16 = use_fp16

    def label_train_dataset(self, imgs_per_window: Subset, ground_truths: List[int], start_time: int) -> TensorDataset:
        set_reproducibility(self.seed)

        cnt = torch.unique(torch.tensor(imgs_per_window.targets), return_counts=True)
        min_cnt = min(cnt[1]).item()
        min_cnt = min(min_cnt, self.num_train_data // len(cnt[1]))
                
        front_indices = []
        back_indices = []
        curr_idx = 0
        for i in range(len(cnt[0])):   
            front_indices = front_indices + [j for j in range(curr_idx, curr_idx + min_cnt)]
            back_indices = back_indices + [j for j in range(curr_idx + min_cnt, curr_idx + cnt[1][i].item())]
            curr_idx = curr_idx + cnt[1][i].item()
        
        random_back_indices = torch.randint(0, len(back_indices), (self.num_train_data - len(front_indices),))
        
        indices = torch.unique(torch.cat((torch.IntTensor(front_indices), torch.IntTensor(back_indices)[random_back_indices]), dim = 0))
        
        train_dataset = Subset(imgs_per_window, indices[0:self.num_train_data])
                
        data_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=16,
            pin_memory=True
        )

        train_inputs = []
        train_targets = []

        data_cnt_per_label = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
        }

        iter_cnt = 0
        self.model.eval()
        total_iterations = len(data_loader)
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to("cuda", non_blocking=True)
                labels = labels.to("cuda", non_blocking=True)
                
                with autocast(enabled=self.use_fp16):
                    outputs = self.model(inputs)

                inputs, outputs = inputs.cpu(), labels.cpu()

                output_cnt = inputs.shape[0]
                for i in range(output_cnt):
                    train_inputs.append(inputs[i])
                    targets = outputs
                    train_targets.append(targets)

                    target_label = targets.item()
                    data_cnt_per_label[target_label] += 1

                iter_cnt += 1
                iter_time = time.time_ns()
                exec_time = (iter_time - start_time) / 1000_000_000

                if exec_time >= self.window_time:
                    break

        train_inputs = torch.stack(train_inputs)
        train_targets = torch.stack(train_targets).squeeze(dim=-1)
        dataset = TensorDataset(train_inputs.cuda(), train_targets.cuda())

        results = {
            "dataset": dataset,
            "total_iterations": total_iterations,
            "processed_iterations": iter_cnt,
            "data_cnt_per_label": data_cnt_per_label
        }

        return results