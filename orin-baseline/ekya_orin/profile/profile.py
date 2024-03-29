from torchvision import models
import torch
import time
import tqdm
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import src.boreas as boreas
from torch.utils.data import Dataset, DataLoader
import src.utils as utils

num_classes = 4
batch_size = 32

def inference(name, model, dataset : Dataset):
    start_time = time.time()
        
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        shuffle=False,
        num_workers=utils.ORIN_CONFIGURES[utils.ORIN_MODE]["cpu"],
        pin_memory=True
    )
    
    model = models.__dict__[name](num_classes=num_classes).to("cuda")
    model.eval()
    corrects = 0
        
    with torch.cuda.amp.autocast():
        for inputs, labels in tqdm.tqdm(data_loader, total=len(data_loader)):
            inputs = inputs.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)
            
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                num_corrects = torch.sum(preds == labels.data)
                corrects += num_corrects

    end_time = time.time() - start_time
    per_img = end_time / 3600

    print(f"{name}-inference_per_img: {per_img}")
    return per_img

def train(name, model, dataset : Dataset):
    start_time = time.time()
    
    model = models.__dict__[name](num_classes=4).to("cuda")
    model.train()
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=utils.ORIN_CONFIGURES[utils.ORIN_MODE]["cpu"],
        pin_memory=True
    )
    
    learning_rate = 0.001
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)        
    
    scaler = torch.cuda.amp.GradScaler()
    
    for inputs, labels in tqdm.tqdm(data_loader, total=len(data_loader)):
        inputs = inputs.to("cuda", non_blocking=True)
        labels = labels.to("cuda", non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    end_time = (time.time() - start_time)
    
    print(f"{name}-train_per_batch:{end_time / len(data_loader)}")
    return end_time / len(data_loader)

def run(stu_name, tea_name):    
    print(f"ORIN: {utils.ORIN_MODE}")

    config, student_model, teacher_model = utils.init_setting(
        "./src/config.json", (
            stu_name,
            f"/home/yskim/experiment/well_trained/{stu_name}.pth",
        ), (
            tea_name,
            f"/home/yskim/experiment/well_trained/{tea_name}.pth",
        )
    )

    scene = "cloudy-rainy-sunny-snowy-cloudy-rainy"
    windows, _ = boreas.generate_boreas_windows(f"/mnt/ext/wukim/datasets/scenario-for-emulator/version-0/{scene}.json")

    i = inference(stu_name, student_model, windows[0])
    t = train(stu_name, student_model, windows[0])
    l = inference(tea_name, teacher_model, windows[0])
    
    return (i,t,l)
    
if __name__ == "__main__":
    utils.ORIN_MODE = "30W"
    
    stu_name = [
        "resnet18",
        "resnet34",
        "vit_b_32"
    ]
           
    tea_name = [
        "wide_resnet50_2",
        "wide_resnet101_2",
        "vit_b_16"
    ]
    
    scene = [
        "cloudy-rainy-sunny-snowy-cloudy-rainy", # S1
        "snowy-rainy-sunny-cloudy-snowy-rainy", # S2
        "sunny-snowy-cloudy-rainy-sunny-snowy", # S3
        "cloudy-rainy-sunny-human-sunny-snowy", # S4
        "rainy-sunny-human-sunny-snowy-cloudy", # S5
        "sunny-human-sunny-cloudy-rainy-snowy", # S6
        'cloudy-sunny-H-H-NH-NH-NH-NH', # S7
        "cloudy-sunny-H-NH-NH-H-NH-NH", # S8
    ]
    
    stu_writer = csv.writer(open(f"{utils.ORIN_MODE}-student.csv", 'w'))
    stu_writer.writerow(["model", "inference", "train"])
    
    tea_writer = csv.writer(open(f"{utils.ORIN_MODE}-teacher.csv", 'w'))
    tea_writer.writerow(["model", "inference"])
    
    for idx in range(3):
        i,t,l = run(stu_name[idx], tea_name[idx])
        
        stu_writer.writerow([stu_name[idx], i, t])
        tea_writer.writerow([tea_name[idx], l])     
        
