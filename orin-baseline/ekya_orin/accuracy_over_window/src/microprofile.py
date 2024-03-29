from torchvision import models
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .utils import EXAMPLE_TRAIN_CONFIGURES, ORIN_CONFIGURES, ORIN_MODE

import torch.nn as nn
import torch.optim as optim
import copy
import time
import torch

from .boreas import generate_boreas_windows

class microprofiler:
    def __init__(self, debug : bool = False):
        self.debug = debug
    
    def set_train_config(self, tr_config : dict):
        self.model_name = tr_config["model_name"]
        self.num_classes = tr_config["num_classes"]
        self.num_hidden_layer = tr_config["num_hidden_layer"]
        self.train_batch_size = tr_config["train_batch_size"]
        self.learning_rate = tr_config["learning_rate"]
        self.momentum = tr_config["momentum"]
        self.microprofile_epochs = tr_config["microprofile_epochs"]
        self.sample_rate = tr_config["sample_rate"]
    
    def set_model(self, model : nn.Module):
        self.model = copy.deepcopy(model).cuda()
        
    def set_dataset(self, dataset : Dataset):        
        dataset_size = len(dataset)
        sample_size = int(dataset_size * self.sample_rate)
        train_size = int(sample_size * 0.8)
        val_size = sample_size - train_size

        if train_size <= 0 or val_size <= 0:
            self.train_indices = None
            self.val_indices = None
        else:   
            indices = torch.randperm(dataset_size)
        
            self.dataset = dataset
            self.train_indices = indices[:train_size]
            self.val_indices = indices[train_size:sample_size]
        
    def profile(self):
        before_train_acc = self.val().item()
        train_time = self.train()
        after_train_acc = self.val().item()
        
        profile_result = {
            "before_train_acc": before_train_acc,
            "train_time_per_epoch": train_time / self.microprofile_epochs,
            "after_train_acc": after_train_acc,
        }
        
        return profile_result
    
    def train(self):
        if self.train_indices is None: return torch.tensor(0.0001)

        if type(self.dataset) is not TensorDataset: self.dataset.dataset.train()
        train_loader = DataLoader(
            Subset(self.dataset, self.train_indices),
            batch_size=self.train_batch_size,
            drop_last=False,
            shuffle=True,
            pin_memory=True,
            num_workers=ORIN_CONFIGURES[ORIN_MODE]["cpu"]
        ) if len(self.train_indices) != 0 else None
        
        start_time = time.time()
        
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        
        scaler = torch.cuda.amp.GradScaler()
        
        for _ in range(self.microprofile_epochs):
            for inputs, labels in train_loader:
                inputs = inputs.to("cuda")
                labels = labels.to("cuda")
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        train_time = time.time() - start_time
        if self.debug: print(f"DEBUG: train time-{train_time}")
        
        return train_time
    
    def val(self):
        if self.val_indices is None: return torch.tensor(0)

        if type(self.dataset) is not TensorDataset: self.dataset.dataset.val()
        val_loader = DataLoader(
            Subset(self.dataset, self.val_indices),
            batch_size=self.train_batch_size,
            drop_last=False,
            shuffle=True,
            pin_memory=True,
            num_workers=ORIN_CONFIGURES[ORIN_MODE]["cpu"]  
        ) if len(self.val_indices) != 0 else None
        
        self.model.eval()
        corrects = 0
        cnts = 0
        
        with torch.cuda.amp.autocast():
            for inputs, labels in val_loader:
                inputs = inputs.to("cuda")
                labels = labels.to("cuda")
                
                with torch.no_grad():
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    num_corrects = torch.sum(preds == labels.data)
                    corrects += num_corrects
                    cnts += len(labels)
                    
        acc = corrects / cnts
        if self.debug: print(f"DEBUG: validation acc-{acc}")
        return acc        
        
if __name__ == "__main__":
    tr_config = copy.deepcopy(EXAMPLE_TRAIN_CONFIGURES)
    tr_config["model_name"] = "resnet18"
    tr_config["sample_rate"] = 1
    
    model = models.__dict__["resnet18"](weights = models.__dict__["ResNet18_Weights"].DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, tr_config["num_hidden_layer"])
    model = nn.Sequential(model, nn.Linear(tr_config["num_hidden_layer"], tr_config["num_classes"]))
    model = model.to("cuda")
    
    windows, ground_truths = generate_boreas_windows("/external-volume/dataset/boreas/boreas/scenario-for-emulator/version-0/sunny-snowy-cloudy-rainy-sunny-snowy.json")
    
    microprofiler_instance = microprofiler()
    microprofiler_instance.set_train_config(tr_config)
    microprofiler_instance.set_model(model)
    microprofiler_instance.set_dataset(windows[0])
    
    profile_result = microprofiler_instance.profile()
    
    print(profile_result)
