import csv
import time
import torch
import gc
import copy

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
    config, student_model, teacher_model = utils.init_setting(
        f"{pwd_path}/ekya_orin/accuracy_over_window/src/config.json", (
            stu_name,
            "",
        ), (
            tea_name,
            "",
        )
    )
    
    student_profile_path = f"{profile_path}/student.csv"
    teacher_profile_path = f"{profile_path}/teacher.csv"
       
    inference_config = utils.gen_inference_config(stu_name, student_profile_path, config["input_stream_fps"])
    train_config = utils.gen_train_config(stu_name, student_profile_path)
    label_config = utils.gen_label_config(tea_name, teacher_profile_path)
    
    output_dir_path = Path(f"{log_path}/{stu_name}/output/{scene}")
    output_dir_path.mkdir(parents=True, exist_ok=True)

    output_file_name = utils.gen_file_name(scene, stu_name, tea_name, config)
    f = open(output_dir_path / output_file_name, "w")
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerow(["name", "acc", "sampling rate", "epochs", "retrained_time"])
    
    profiler = microprofile.microprofiler()
    windows = bdd.generate_bdd_windows(f"{dataset_path}/{scene}", config["window_time"])
    train_data_from_prev_window = None
    
    epochs_list, sample_rates_list = utils.gen_range_sample_rate_and_epochs(inference_config, train_config, label_config, config)
        
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
        
        init_time = time.time()
        
        acc = func.inference(student_model, windows[window_idx], best_config_set["inference"])
        train_data_from_prev_window = func.labeling(teacher_model, windows[window_idx], best_config_set["label"], False)
        tmp_time = time.time()
        func.train(student_model, train_data_from_prev_window, best_config_set["train"], (init_time, config["window_time"]))
        retrained_time = time.time() - tmp_time
        
        torch.cuda.empty_cache()
        gc.collect()
        
        csv_writer.writerow([f"window #{window_idx}",
                             acc * 100.,
                             best_config_set["label"]["sample_rate"],
                             best_config_set["train"]["epochs"],
                             retrained_time])
    f.close()
    shutil.copyfile(src=output_dir_path / output_file_name,
                    dst=output_dir_path / "result.csv")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mode", type=str)
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
    
    print(utils.ORIN_MODE)
    
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
    
    bdd_scene = [
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6"
    ]
    
    for sss in range(len(bdd_scene)):
       for iii in range(len(stu_name)):
           run(stu_name[iii], tea_name[iii], bdd_scene[sss])
