import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import time
import math
import tqdm

from torch.utils.data import Dataset, Subset, TensorDataset
from torch.utils.data.dataloader import DataLoader
from .utils import ORIN_CONFIGURES, ORIN_MODE

def inference(student_model, window, inference_config):
    window.val()
    data_loader = DataLoader(
        window,
        batch_size=1,
        drop_last=False,
        shuffle=False,
        num_workers=ORIN_CONFIGURES[ORIN_MODE]["cpu"],
        pin_memory=True
    )

    student_model.eval()
    corrects = 0
    
    with torch.cuda.amp.autocast():
        for inputs, labels in tqdm.tqdm(data_loader, total=len(data_loader)):
            inputs = inputs.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)
            
            with torch.no_grad():
                outputs = student_model(inputs)
                _, preds = torch.max(outputs, 1)
                
                num_corrects = torch.sum(preds == labels.data)
                corrects += num_corrects
    
    acc = corrects / len(data_loader)
    return acc.item()

def sample_data_to_train(dataset: Dataset, sampling_rate: float) -> Dataset:
    total_num = len(dataset)
    train_num_imgs = int(math.floor(total_num * sampling_rate))
    
    indices = np.arange(0, total_num, total_num / train_num_imgs).round(0).astype(int).tolist()
    
    cnt = torch.unique(torch.tensor(dataset.targets)[indices], return_counts = True)
    dist = (cnt[1] / torch.sum(cnt[1])).tolist()
    
    for i, d in enumerate(dist):
        print(f"label {cnt[0][i].item()} images: {int(d * train_num_imgs)}")
    
    train_dataset = Subset(dataset, indices)
    print(f"# of images: {total_num}, sampled # of images: {len(train_dataset)}")        
    return train_dataset

def train(student_model, prev_window, train_config, time_info):
    if prev_window is None:
        return
       
    init_time, window_time = time_info
    
    if type(prev_window) is not TensorDataset:
        prev_window.dataset.train()
    
    data_loader = DataLoader(
        prev_window,
        batch_size=train_config["train_batch_size"],
        drop_last=False,
        shuffle=False,
        num_workers=ORIN_CONFIGURES[ORIN_MODE]["cpu"],
        pin_memory=True
    )
    
    student_model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(student_model.parameters(), lr=train_config["learning_rate"], momentum=train_config["momentum"])

    scaler = torch.cuda.amp.GradScaler()
    
    for _ in tqdm.tqdm(range(train_config["epochs"]), total=train_config["epochs"]):
        for inputs, labels in data_loader:
            inputs = inputs.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = student_model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if time.time() - init_time > window_time:
                return
    
def labeling(teacher_model, window, label_config, teacher_labeled = False):
    sampled_window = sample_data_to_train(window, label_config["sample_rate"])
    sampled_window.dataset.val()
    
    data_loader = DataLoader(
        sampled_window,
        batch_size=1,
        drop_last=False,
        shuffle=False,
        num_workers=ORIN_CONFIGURES[ORIN_MODE]["cpu"],
        pin_memory=True
    )

    teacher_model.eval()
    
    train_inputs = []
    train_targets = []
    
    with torch.cuda.amp.autocast():
        for inputs, labels in tqdm.tqdm(data_loader, total=len(data_loader)):
            inputs = inputs.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)
            
            with torch.no_grad():
                outputs = teacher_model(inputs)
                _, preds = torch.max(outputs, 1)
                
            train_inputs.append(inputs.cpu())
            train_targets.append(preds.cpu())
    
    train_inputs = torch.stack(train_inputs)
    tmp = train_inputs.shape
    train_inputs = train_inputs.reshape((tmp[0], tmp[2], tmp[3], tmp[4]))
    train_targets = torch.stack(train_targets).squeeze(dim=-1)
    
    if teacher_labeled:
        sampled_window = TensorDataset(train_inputs, train_targets)
    
    return sampled_window

        
