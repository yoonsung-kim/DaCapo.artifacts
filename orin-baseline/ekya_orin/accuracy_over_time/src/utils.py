import copy
import csv
import torch.nn as nn
import timm
import torch

from torchvision import models
from .src.util.reprod_util import set_reproducibility
from .src.config.config import Config

ORIN_MODE = "default"

def set_mode(mode):
    global ORIN_MODE
    ORIN_MODE = mode

# train config format
EXAMPLE_TRAIN_CONFIGURES = {
    "model_name": "resnet18",
    "num_classes": 9,
    "train_batch_size": 32,
    "learning_rate": 0.0001,
    "momentum": 0.9,
    "num_hidden_layer": 64,
    "microprofile_epochs": 5,
}

ORIN_CONFIGURES = {
    "default": {
        "cpu": 12
    },
    "50W": {
        "cpu": 12        
    },    
    "30W": {
        "cpu": 8        
    },
    "15W": {
        "cpu": 4        
    }
}

def convert_new_config_into_old_config(new_config):
    res = dict()
    
    res["seed"] = new_config.seed
    res["num_classes"] = new_config.num_classes
    res["window_time"] = new_config.window_time
    res["input_stream_fps"] = new_config.fps
    res["train_data_sampling_rate"] = new_config.initial_sampling_rate
    res["train_epochs"] = new_config.initial_epoch
    res["train_batch_size"] = new_config.train_batch_size
    res["inference_batch_size"] = new_config.infer_batch_size
    res["teacher_batch_size"] = new_config.infer_batch_size
    res["num_imgs_per_window"] = new_config.window_time * new_config.fps
    res["use_fp16"] = True
    res["update_weight_within_window"] = True
    res["lr"] = new_config.lr
    res["use_bfp"] = False

    return res

def gen_inference_config(stu_name, file_name, FPS=30):
    inference_config = dict()
    inference_config["FPS"] = FPS
    inference_config["model_name"] = stu_name
    
    reader = csv.reader(open(file_name))
    for row in reader:
        if row[0] == stu_name:
            if row[1] == ORIN_MODE:
                inference_config["resource"] = float(row[2])
    
    
    return inference_config

def gen_train_config(stu_name, file_name):
    train_config = copy.deepcopy(EXAMPLE_TRAIN_CONFIGURES)
    train_config["model_name"] = stu_name
    
    reader = csv.reader(open(file_name))
    for row in reader:
        if row[0] == stu_name:
            if row[1] == ORIN_MODE:
                train_config["resource"] = float(row[3])
                
    return train_config

def gen_label_config(tea_name, file_name):
    label_config = dict()
    label_config["model_name"] = tea_name
    
    reader = csv.reader(open(file_name))
    for row in reader:
        if row[0] == tea_name:
            if row[1] == ORIN_MODE:
                label_config["resource"] = float(row[2])
    
    return label_config

def gen_file_name(scene, stu, tea, config):
    name = f"{scene}-{stu}-{tea}-" \
                   f"w{config['window_time']}-fps{config['input_stream_fps']}-{ORIN_MODE}"
                   
    return f"output-{name}.csv"

def tmp_convert(model, num_classes):
    last_layer = model.head
    model.head = nn.Linear(
        last_layer.in_features,
        num_classes,
        bias = True
    )
    
    return model

def init_setting(config_path, student_model_info, teacher_model_info):
    set_reproducibility(128)
    
    cl_config = Config(config_path)
    config = convert_new_config_into_old_config(cl_config)
    
    student_model_name, student_model_path = student_model_info
    
    if student_model_name == "vit_b_32":
        student_model = timm.create_model('vit_base_patch32_224.augreg_in1k', pretrained=True)
        student_model = tmp_convert(student_model, config["num_classes"])
    else:
        student_model = models.__dict__[student_model_name](pretrained=True)
        student_model.fc = nn.Linear(student_model.fc.in_features, config["num_classes"])      

    student_model.cuda()
    
    teacher_model_name, teacher_model_path = teacher_model_info
    
    if teacher_model_name == "vit_b_16":
        teacher_model = timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=True)
        teacher_model = tmp_convert(teacher_model, config["num_classes"])
    else:
        teacher_model = models.__dict__[teacher_model_name](pretrained=True)
        teacher_model.fc = nn.Linear(teacher_model.fc.in_features, config["num_classes"])      
    
    teacher_model.cuda()
    teacher_model.load_state_dict(torch.load(teacher_model_path, map_location="cuda:0")["model"])
    
    return config, student_model, teacher_model

def gen_range_sample_rate_and_epochs(inference_config, train_config, label_config, config):    
    inference_res = inference_config["resource"]
    train_res = train_config["resource"]
    label_res = label_config["resource"]
    
    total_imgs = config["window_time"] * config["input_stream_fps"]
    
    epochs_list = [5, 15, 30]
    epochs_max_rate = [0, 0, 0]
    
    for idx, epochs in enumerate(epochs_list):
        tmp_rate = 0
        
        for rate in range(1,101):
            sample_rate = rate / 100
            labeled_imgs = int(total_imgs * sample_rate)
            
            required_time = total_imgs * inference_res
            required_time += ((labeled_imgs + 31) // 32) * train_res * epochs
            required_time += labeled_imgs * label_res
            
            if required_time > config["window_time"]: continue
            tmp_rate = rate
        
        epochs_max_rate[idx] = tmp_rate

    low_rate = min(epochs_max_rate)
    low_rate = max(low_rate, 1)
    
    high_rate = max(epochs_max_rate)
    high_rate = max(high_rate, 1)
        
    middle_rate = (low_rate + high_rate) // 2
    
    low_rate = low_rate / 100
    middle_rate = middle_rate / 100
    high_rate = high_rate / 100
        
    sample_rates_list = [low_rate, middle_rate, high_rate]
    
    return epochs_list, sample_rates_list
