import csv
import time
import torch
import gc
import copy
import json

import src.thief_scheduler as thief_scheduler
import src.microprofile as microprofile
import src.bdd as bdd
import src.utils as utils
import src.func as func

import shutil
from pathlib import Path

pwd_path = ""
dataset_path = ""
weight_path = ""
profile_path = ""
log_path = ""

def run(stu_name, tea_name, scene):
    print(f"ORIN: {utils.ORIN_MODE}")
    
    config, student_model, teacher_model = utils.init_setting(
        f"{pwd_path}/ekya_orin/accuracy_over_time/src/config.json", (
            stu_name,
            f"{weight_path}/{stu_name}.pth",
        ), (
            tea_name,
            f"{weight_path}/{tea_name}.pth",
        )
    )
    
    student_profile_path = f"{profile_path}/student.csv"
    teacher_profile_path = f"{profile_path}/teacher.csv"   
    
    inference_config = utils.gen_inference_config(stu_name, student_profile_path, config["input_stream_fps"])
    train_config = utils.gen_train_config(stu_name, student_profile_path)
    label_config = utils.gen_label_config(tea_name, teacher_profile_path)
        
    profiler = microprofile.microprofiler()
    windows = bdd.generate_bdd_windows(f"{dataset_path}/{scene}", config["window_time"])
    train_data_from_prev_window = None
    
    epochs_list, sample_rates_list = utils.gen_range_sample_rate_and_epochs(inference_config, train_config, label_config, config)    
    log_info_list = []

    output_dir_path = Path(f"{log_path}/{stu_name}/output/{scene}")
    output_dir_path.mkdir(parents=True, exist_ok=True)
        
    for window_idx in range(0, len(windows)):
        profile_res_list = list()
        
        if train_data_from_prev_window != None:
            for rate in sample_rates_list:
                tmp_train_config = copy.deepcopy(train_config)
                tmp_train_config["sample_rate"] = rate
                
                profiler.set_train_config(tmp_train_config)
                profiler.set_model(student_model)
                profiler.set_dataset(train_data_from_prev_window)
                
                profile_res = profiler.profile()                
                profile_res_list.append((tmp_train_config, profile_res))
        
        config_set_list = list()
        for train_config, profile_res in profile_res_list:
            for epochs in epochs_list:
                tmp_train_config = copy.deepcopy(train_config)
                tmp_train_config["epochs"] = epochs
                
                tmp_label_config = copy.deepcopy(label_config)
                tmp_label_config["sample_rate"] = tmp_train_config["sample_rate"]
                
                config_set = {
                    "train": tmp_train_config,
                    "inference": inference_config,
                    "label": tmp_label_config,
                    "profile": profile_res,
                }
                
                config_set_list.append(config_set)
                
        best_config_set = None

        if len(config_set_list) > 0:
            best_config_set = thief_scheduler.thief_scheduler(config_set_list, config["window_time"])
        
        if best_config_set is None:
            tmp_train_config = copy.deepcopy(train_config)
            tmp_train_config["sample_rate"] = sample_rates_list[0]
            tmp_train_config["epochs"] = epochs_list[0]
                
            tmp_label_config = copy.deepcopy(label_config)
            tmp_label_config["sample_rate"] = sample_rates_list[0]
            
            best_config_set = {
                "train": tmp_train_config,
                "inference": inference_config,
                "label": tmp_label_config,
            }
                
        before_student_model = copy.deepcopy(student_model)
        init_time = time.time()
        
        acc, _ = func.inference((before_student_model, student_model), windows[window_idx], best_config_set["inference"], (config["window_time"], config["window_time"] * 2))
        inference_time = time.time() - init_time
        
        train_data_window = func.labeling(teacher_model, windows[window_idx], best_config_set["label"], False)
        labeling_time = (time.time() - init_time) - inference_time
        
        func.train(student_model, train_data_from_prev_window, best_config_set["train"], (init_time, config["window_time"]))
        retrained_time = (time.time() - init_time) - (inference_time + labeling_time)
        real_retrained_time = retrained_time * (120.0 / (120.0 - inference_time))
                
        acc, log_info = func.inference((before_student_model, student_model), windows[window_idx], best_config_set["inference"], (config["window_time"], real_retrained_time))

        log_info_list.append({"retrain_time" : real_retrained_time, "log_info" : log_info})

        train_data_from_prev_window = train_data_window
        
        torch.cuda.empty_cache()
        gc.collect()

    with open(output_dir_path / f"student_file-{scene}-{stu_name}-{tea_name}.json", "w") as json_file:
        json.dump(log_info_list, json_file) 

    shutil.copyfile(src=output_dir_path / f"student_file-{scene}-{stu_name}-{tea_name}.json",
                    dst=output_dir_path / "result.json")  

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mode", type=str)
    parser.add_argument("--student", type=str)
    parser.add_argument("--teacher", type=str)
    parser.add_argument("--scene", type=str)
    parser.add_argument("--pwd_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--weight_path", type=str)
    parser.add_argument("--profile_path", type=str)
    parser.add_argument("--log_path", type=str)

    args = parser.parse_args()

    pwd_path = args.pwd_path
    dataset_path = args.dataset_path
    weight_path = args.weight_path
    profile_path = args.profile_path
    log_path = args.log_path
    
    utils.set_mode(args.mode)
    run(args.student, args.teacher, args.scene)