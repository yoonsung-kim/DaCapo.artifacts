import os
import csv
import copy
import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="summarize results for figure. 10")
parser.add_argument("--output-root", type=str, help="root path of output")
parser.add_argument("--summary-root", type=str, default="./", help="root path of summarized results")


NUM_WINDOWS = 10
NUM_WARMUP = 4
FPS = 30
WINDOW_TIME = 120
WINDOW_TIME_EOMU = 10
NUM_IMGS_PER_WINDOW = FPS * WINDOW_TIME

NAME_DICT = {
    "orin_low-ekya": "OrinLow-Ekya",
    "orin_high-ekya": "OrinLow-Ekya",
    "dacapo-ekya": "DaCapo-Ekya",
    "eomu": "OrinHigh-EOMU",
    "spatial": "DaCapo-Spatial",
    "spatiotemporal": "DaCapo-Spatiotemporal",
    "fp": "RTX3090-CL"
}


def prepare_dc(name_to_save: str, dict, dst_root: Path):
    # >>>>> DS-ST >>>>>
    dc_st = dict["dc-st"]
    if not os.path.isfile(dc_st["ori-correct"]) or \
        not os.path.isfile(dc_st["ori-end_point"]):
        return
    
    with open(dc_st["ori-correct"]) as f:
        data = json.load(f)
        corrects = []
        for row in data:
            corrects.extend(row)
        corrects = np.array(corrects).flatten()
        corrects = corrects[NUM_IMGS_PER_WINDOW*NUM_WARMUP:]
        corrects = corrects.reshape(10, 12, 300).tolist()

    file_path = dst_root / f"{name_to_save}_dc_st_correct.json"
    dc_st["correct"] = str(file_path)
    with open(file_path, "w") as f:
        json.dump(corrects, f, indent=4)

    times = []
    with open(dc_st["ori-end_point"]) as f:
        csv_reader = csv.reader(f)
        next(csv_reader)

        acc_images = 0
        for row in csv_reader:
            phase_name, images = row[1], int(row[2])

            acc_images += images
            if "train" in phase_name:
                times.append(acc_images / FPS)
    file_path = dst_root / f"{name_to_save}_dc_st_end_point.json"
    dc_st["end_point"] = str(file_path)
    with open(file_path, "w") as f:
        json.dump(times, f, indent=4)
    # <<<<< DS-ST <<<<<

    # >>>>> DS-S >>>>>
    dc_s = dict["dc-s"]
    if not os.path.isfile(dc_s["ori-correct"]) or \
        not os.path.isfile(dc_s["ori-end_point"]):
        return
    
    with open(dc_s["ori-correct"]) as f:
        data = json.load(f)
        corrects = []
        for key in data.keys():
            corrects.extend(data[key])
        corrects = np.array(corrects).flatten()
        corrects = corrects[NUM_IMGS_PER_WINDOW*NUM_WARMUP:]
        corrects = corrects.reshape(10, 12, 300).tolist()

    file_path = dst_root / f"{name_to_save}_dc_s_correct.json"
    dc_s["correct"] = str(file_path)
    with open(file_path, "w") as f:
        json.dump(corrects, f, indent=4)

    times = []
    with open(dc_s["ori-end_point"]) as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        next(csv_reader)

        acc_images = 0
        for row in csv_reader:
            int(row[10])
            acc_images += images
            times.append(acc_images / FPS)
    file_path = dst_root / f"{name_to_save}_dc_s_end_point.json"
    dc_s["end_point"] = str(file_path)
    with open(file_path, "w") as f:
        json.dump(times, f, indent=4)
    # <<<<< DS-S <<<<<


def prepare_eomu(name_to_save: str, dict, dst_root: Path):
    eomu = dict["eomu"]
    if not os.path.isfile(eomu["ori-correct"]) or \
        not os.path.isfile(eomu["ori-end_point"]):
        return
    
    with open(eomu["ori-correct"]) as f:
        data = json.load(f)
        corrects = []
        for row in data:
            corrects.extend(row)
        corrects = np.array(corrects).flatten().tolist()

    file_path = dst_root / f"{name_to_save}_eomu_correct.json"
    eomu["correct"] = str(file_path)
    with open(file_path, "w") as f:
        json.dump(corrects, f, indent=4)

    times = []
    with open(eomu["ori-end_point"]) as f:
        csv_reader = csv.reader(f)
        next(csv_reader)

        for row in csv_reader:
            window_idx = int(row[0])
            times.append((window_idx + 1) * 10)
    file_path = dst_root / f"{name_to_save}_eomu_end_point.json"
    eomu["end_point"] = str(file_path)
    with open(file_path, "w") as f:
        json.dump(times, f, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    output_root = Path(args.output_root)
    summary_root = Path(args.summary_root)

    fig10_dir = summary_root / "figure_10"
    tmp_dir = fig10_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    template = {
        "dc-st": {"ori-correct": "", "ori-end_point": "", "correct": "", "end_point": ""},
        "dc-s": {"ori": "", "correct": "", "end_point": ""},
        "eomu": {"ori": "", "correct": "", "end_point": ""},
        "ekya": {"ori": "", "correct": "", "end_point": ""}
    }

    # >>>>> file init >>>>>
    model = "resnet18"
    resnet18_files = copy.deepcopy(template)

    cl_type = "spatiotemporal"
    resnet18_files["dc-st"]["ori-correct"] = \
        output_root / cl_type / model / "output" / "s1" / "result.json"
    resnet18_files["dc-st"]["ori-end_point"] = \
        output_root / cl_type / model / "output" / "s1" / "phase_log.csv"
    
    cl_type = "spatial"
    resnet18_files["dc-s"]["ori-correct"] = \
        output_root / cl_type / model / "output" / "s1" / "result.json"
    resnet18_files["dc-s"]["ori-end_point"] = \
        output_root / cl_type / model / "output" / "s1" / "result.csv"
    prepare_dc(name_to_save=f"{model}", dict=resnet18_files, dst_root=tmp_dir)
    
    cl_type = "eomu"
    resnet18_files["eomu"]["ori-correct"] = \
        output_root / cl_type / model / "output" / "s1" / "result.json"
    resnet18_files["eomu"]["ori-end_point"] = \
        output_root / cl_type / model / "output" / "s1" / "train_log.csv"
    prepare_eomu(name_to_save=f"{model}", dict=resnet18_files, dst_root=tmp_dir)
    
    cl_type = "orin_high-ekya"
    resnet18_files["ekya"]["correct"] = \
        output_root / "orin_high-ekya" / model / "output" / "s1" / "result.json"
    resnet18_files["ekya"]["end_point"] = \
        output_root / "orin_high-ekya" / model / "output" / "s1" / "result.json"
    
    if not os.path.isfile(resnet18_files["ekya"]["correct"]) or \
        not os.path.isfile(resnet18_files["ekya"]["end_point"]):
        resnet18_files["ekya"]["correct"] = ""
        resnet18_files["ekya"]["end_point"] = ""

    model = "resnet34"
    resnet34_files = copy.deepcopy(template)

    cl_type = "spatiotemporal"
    resnet34_files["dc-st"]["ori-correct"] = \
        output_root / cl_type / model / "output" / "s1" / "result.json"
    resnet34_files["dc-st"]["ori-end_point"] = \
        output_root / cl_type / model / "output" / "s1" / "phase_log.csv"
    
    cl_type = "spatial"
    resnet34_files["dc-s"]["ori-correct"] = \
        output_root / cl_type / model / "output" / "s1" / "result.json"
    resnet34_files["dc-s"]["ori-end_point"] = \
        output_root / cl_type / model / "output" / "s1" / "result.csv"
    prepare_dc(name_to_save=f"{model}", dict=resnet34_files, dst_root=tmp_dir)
    
    cl_type = "eomu"
    resnet34_files["eomu"]["ori-correct"] = \
        output_root / cl_type / model / "output" / "s1" / "result.json"
    resnet34_files["eomu"]["ori-end_point"] = \
        output_root / cl_type / model / "output" / "s1" / "train_log.csv"
    prepare_eomu(name_to_save=f"{model}", dict=resnet34_files, dst_root=tmp_dir)
    
    cl_type = "orin_high-ekya"
    resnet34_files["ekya"]["correct"] = \
        output_root / "orin_high-ekya" / model / "output" / "s1" / "result.json"
    resnet34_files["ekya"]["end_point"] = \
        output_root / "orin_high-ekya" / model / "output" / "s1" / "result.json"
    
    if not os.path.isfile(resnet34_files["ekya"]["correct"]) or \
        not os.path.isfile(resnet34_files["ekya"]["end_point"]):
        resnet34_files["ekya"]["correct"] = ""
        resnet34_files["ekya"]["end_point"] = ""
    # <<<<< file init <<<<<

    sigma = 0.2
    fps = 30
    window_time = 120
    total_window_cnt = 14
    option = 0

    start_window = 0
    window_cnt = 10
    step = 40
    unit_sec = 10
    shift_left = 0

    x_ticks, y_ticks = 60, 10
    start_y, end_y = 60, 100
    width, height, dpi = 800, 200, 100
    fontsize = 18
    linewidth = 2
    dotsize = 20

    # src files
    mystique_file = ""
    static_file = ""
    mm_file = ""
    orin_file = ""

    # dst file
    save_file = ""

    start_range = start_window * window_time * fps
    end_range = start_range + window_cnt * window_time * fps

    def RGB(r, g, b): return f"#{format((r << 16) | (g << 8) | b, '06X')}"

    weak_blue = RGB(118, 172, 255)
    strong_blue = RGB(42, 78, 170)
    weak_black = RGB(200, 200, 200)
    strong_black = RGB(80, 80, 80)

    def read_dacapo(corrects_file_name, end_of_train_points_file_name, step = 1):
        # parse
        raw_data = list()
        
        with open(corrects_file_name, 'r') as f:
            j = json.load(f)
            
            for json_set in j:  
                for json_list in json_set:
                    raw_data += json_list

        raw_avg_data = list()
        
        for i in range(1, len(raw_data) // (unit_sec * fps) + 1):
            raw_avg_data.append((i * unit_sec, sum(raw_data[(i - 1) * unit_sec * fps : i * unit_sec * fps]) / (unit_sec * fps) * 100))
            
        x, y = list(), list()
        
        for r_x, r_y in raw_avg_data:
            if len(x) == 0:
                x.append(0)
                y.append(r_y)
            else:
                x.append(r_x)
                y.append(r_y)
                
        dx, dy = list(), list()
        
        try:
            with open(end_of_train_points_file_name, 'r') as f:
                json_list = json.load(f)
                
                for point in json_list:
                    x_target = int(point - 4 * window_time)
                    
                    for i in range(1, len(raw_avg_data)):
                        s_i, s_v = raw_avg_data[i-1]
                        e_i, e_v = raw_avg_data[i]
                        
                        if s_i <= x_target and x_target <= e_i:
                            t_len = e_i - s_i
                            s_r = (x_target - s_i) / t_len
                            e_r = (e_i - x_target) / t_len
                            
                            val = s_v * e_r + e_v * s_r
                            
                            dx.append(x_target)
                            dy.append(val)
        except:
            pass
                        
        for i in range(len(x)):
            x[i] -= start_range / fps
            
        for i in range(len(dx)):
            dx[i] -= start_range / fps
        
        return x, y, dx, dy


    def read_ekya(corrects_file_name, end_of_train_points_file_name, step = 1):
        # parse
        raw_data = list()
        
        with open(corrects_file_name, 'r') as f:
            j = json.load(f)
            
            for json_set in j:  
                raw_data += json_set["log_info"]

        raw_avg_data = list()
        
        for i in range(1, len(raw_data) // (unit_sec * fps) + 1):
            raw_avg_data.append((i * unit_sec- 4 * window_time, sum(raw_data[(i - 1) * unit_sec * fps : i * unit_sec * fps]) / (unit_sec * fps) * 100))
            
        x, y = list(), list()
        
        for r_x, r_y in raw_avg_data:
            if len(x) == 0:
                x.append(0)
                y.append(r_y)
            else:
                x.append(r_x)
                y.append(r_y)
                
        dx, dy = list(), list()
        
        try:
            with open(end_of_train_points_file_name, 'r') as f:
                j = json.load(f)
                current_time = 0
                for json_set in j:
                    x_target = int(current_time + json_set["retrain_time"] - 4 * window_time)
                    current_time += window_time
                    
                    for i in range(1, len(raw_avg_data)):
                        s_i, s_v = raw_avg_data[i-1]
                        e_i, e_v = raw_avg_data[i]
                        
                        if s_i <= x_target and x_target <= e_i:
                            t_len = e_i - s_i
                            s_r = (x_target - s_i) / t_len
                            e_r = (e_i - x_target) / t_len
                            
                            val = s_v * e_r + e_v * s_r
                            
                            dx.append(x_target)
                            dy.append(val)
        except:
            pass
                        
        for i in range(len(x)):
            x[i] -= start_range / fps
            
        for i in range(len(dx)):
            dx[i] -= start_range / fps
        
        return x, y, dx, dy

    def read_mm(corrects_file_name, end_of_train_points_file_name, step = 1):
        # parse
        raw_data = list()
        
        with open(corrects_file_name, 'r') as f:
            j = json.load(f)
            raw_data += j

        raw_avg_data = list()
        
        for i in range(1, len(raw_data) // (unit_sec * fps) + 1):
            raw_avg_data.append((i * unit_sec- 4 * window_time, sum(raw_data[(i - 1) * unit_sec * fps : i * unit_sec * fps]) / (unit_sec * fps) * 100))
            
        x, y = list(), list()
        
        for r_x, r_y in raw_avg_data:
            if len(x) == 0:
                x.append(0)
                y.append(r_y)
            else:
                x.append(r_x)
                y.append(r_y)
                
        dx, dy = list(), list()
        
        try:
            with open(end_of_train_points_file_name, 'r') as f:
                j = json.load(f)
                for point in j:
                    x_target = int(point - 4 * window_time)
                    
                    for i in range(1, len(raw_avg_data)):
                        s_i, s_v = raw_avg_data[i-1]
                        e_i, e_v = raw_avg_data[i]
                        
                        if s_i <= x_target and x_target <= e_i:
                            t_len = e_i - s_i
                            s_r = (x_target - s_i) / t_len
                            e_r = (e_i - x_target) / t_len
                            
                            val = s_v * e_r + e_v * s_r
                            
                            dx.append(x_target)
                            dy.append(val)
        except:
            pass
                        
        for i in range(len(x)):
            x[i] -= start_range / fps
            
        for i in range(len(dx)):
            dx[i] -= start_range / fps
        
        return x, y, dx, dy

    def draw(drift_time_list, important_drift_time_list):
        plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)

        for drift_time in drift_time_list: plt.axvline(x = drift_time - start_range / fps, color = "lightgray", linestyle="--", zorder = 0)
        for drift_time in important_drift_time_list: plt.axvline(x = drift_time - start_range / fps, color = "grey", linestyle="-", zorder = 0)
        
        if option == 1:
            sss = 0
            eee = window_time * window_cnt
            plt.xlim(0, window_time * window_cnt)
        if option == 2:
            sss = -30
            eee = window_time * window_cnt + 30
            plt.xlim(-30, window_time * window_cnt + 30)
        
        tmp_vline_list = [i - start_range / fps for i in range(0, window_time * total_window_cnt, window_time)]
        vline_list = list()
            
        for vline in tmp_vline_list:
            if vline < 0 or window_time * window_cnt < vline: continue
            vline_list.append(vline)

        tmp = ["" for _ in vline_list]
        
        plt.xticks(vline_list, tmp)
        plt.tick_params(axis='x', labelsize=fontsize)

        plt.ylim(start_y, end_y)
        plt.yticks(range(start_y, end_y + y_ticks, y_ticks))
        plt.tick_params(axis='y', labelsize=fontsize, left=False)

        if mystique_file[0] != "":
            x_mystique, y_mystique, dx_mystique, dy_mystique = read_dacapo(mystique_file[0], mystique_file[1], step = step)
            plt.plot(x_mystique, y_mystique, linestyle='-', color=strong_blue, label="Mystique", linewidth=linewidth, zorder = 7)
            plt.scatter(dx_mystique, dy_mystique, color=strong_blue, edgecolor='black', s=dotsize, zorder = 20, marker="o")

        if static_file[0] != "":
            x_static, y_static, dx_static, dy_static = read_dacapo(static_file[0], static_file[1], step = step)
            plt.plot(x_static, y_static, linestyle='-', color=weak_blue, label="Static Config", linewidth=linewidth, zorder = 5)
            plt.scatter(dx_static, dy_static, color=weak_blue, edgecolor='black', s=dotsize, zorder = 19, marker="o")
        
        if mm_file[0] != "":
            x_mm, y_mm, dx_mm, dy_mm = read_mm(mm_file[0], mm_file[1], step = step)
            plt.plot(x_mm, y_mm, linestyle='-', color=strong_black, label="Static Config", linewidth=linewidth, zorder = 3)
            plt.scatter(dx_mm, dy_mm, color=strong_black, edgecolor='black', s=dotsize, zorder = 18, marker="o")        
        
        if orin_file[0] != "":
            x_orin, y_orin, dx_orin, dy_orin = read_ekya(orin_file[0], orin_file[1], step = step)
            plt.plot(x_orin, y_orin, linestyle='-', color=weak_black, label="Orin Config", linewidth=linewidth, zorder = 1)
            plt.scatter(dx_orin, dy_orin, color=weak_black, edgecolor='black', s=dotsize, zorder = 17, marker="o")
        
        # history = [dict() for i in range(4)]
        # for i in range(len(x_mystique)):
        #     history[0][x_mystique[i]] = y_mystique[i]
            
        # for i in range(len(x_static)):
        #     history[1][x_static[i]] = y_static[i]
            
        # for i in range(len(x_mm)):
        #     history[2][x_mm[i]] = y_mm[i]
        
        # for i in range(len(x_orin)):
        #     history[3][x_orin[i]] = y_orin[i]    
            
        # comparison = [0 for i in range(16)]        
        # for i in range(sss, eee + unit_sec, unit_sec):
        #     for j in range(16):
        #         first = j // 4
        #         second = j % 4
        #         comparison[j] = max(comparison[j], history[first][i] - history[second][i])
        
        # string_set = ["DACAPO-spatiotemporal", "DACAPO-spatial", "EOMU", "EKYA"]
        
        # for i in range(16):
        #     first = i // 4
        #     second = i % 4
            
        #     print(f"MAX difference case: {string_set[first]} > {string_set[second]} : {comparison[i]}")        

        plt.savefig(str(fig10_dir / save_file), dpi=dpi, format="pdf", bbox_inches="tight")
        plt.show()

    ## ResNet18 - WideResNet50 (Scenario #1)
    sigma = 0.2
    fps = 30
    window_time = 120

    step = 30
    linewidth = 3
    dotsize=70

    # mystique_file = ["resnet18-scenario_1-mystique-corrects_per_10_sec.json", ""]
    # static_file = ["resnet18-scenario_1-static-corrects_per_10_sec.json", ""]
    # mm_file = ["corrects-resnet18_lr0.001-scenario_0-type_1-clear-resnet18-wide_resnet50_2.json", ""]

    # mystique_file = ["", ""]
    # static_file = ["", ""]
    # mm_file = ["", ""]
    # orin_file = ["student_file-scenario_0-type_1-clear-resnet18-wide_resnet50_2.json", ""]

    mystique_file = [
        resnet18_files["dc-st"]["correct"],
        ""
    ]
    static_file = [
        resnet18_files["dc-s"]["correct"],
        ""
    ]
    mm_file = [
        resnet18_files["eomu"]["correct"],
        ""
    ]
    orin_file = [
        resnet18_files["ekya"]["correct"],
        ""
    ]

    x_ticks, y_ticks = 120, 25
    start_y, end_y = 50, 100
    width, height, dpi = 800, 80, 100
    fontsize = 15

    # save_file = "accuracy-over-time-resnet18-overall.pdf"
    save_file = "figure_10_a.pdf"

    start_window = 0
    window_cnt = 9
    unit_sec = 15
    shift_left = 60
    option = 1

    start_range = start_window * window_time * fps + shift_left * fps
    end_range = start_range + window_cnt * window_time * fps

    draw([90, 390, 690, 990], [])

    # draw with end points
    mystique_file = [
        resnet18_files["dc-st"]["correct"],
        resnet18_files["dc-st"]["end_point"]
    ]
    static_file = [
        resnet18_files["dc-s"]["correct"],
        resnet18_files["dc-s"]["end_point"]
    ]
    mm_file = [
        resnet18_files["eomu"]["correct"],
        resnet18_files["eomu"]["end_point"]
    ]
    orin_file = [
        resnet18_files["ekya"]["correct"],
        resnet18_files["ekya"]["end_point"]
    ]


    x_ticks, y_ticks = 120, 10
    start_y, end_y = 50, 100
    width, height, dpi = 350, 200, 100
    fontsize = 15

    # save_file = "accuracy-over-time-resnet18-part1.pdf"
    save_file = "figure_10_c.pdf"

    start_window = 1
    window_cnt = 2
    unit_sec = 15
    shift_left = 0
    option = 2

    start_range = start_window * window_time * fps + shift_left * fps
    end_range = start_range + window_cnt * window_time * fps
    draw([], [180, ])

    x_ticks, y_ticks = 120, 10
    start_y, end_y = 50, 100
    width, height, dpi = 350, 200, 100
    fontsize = 15

    # save_file = "accuracy-over-time-resnet18-part2.pdf"
    save_file = "figure_10_d.pdf"

    start_window = 6
    window_cnt = 2
    unit_sec = 15
    shift_left = 0
    option = 2

    start_range = start_window * window_time * fps + shift_left * fps
    end_range = start_range + window_cnt * window_time * fps

    draw([], [720, 840 ])

    ## ResNet34 - WideResNet101 (Scenario #1)
    sigma = 0.2
    fps = 30
    window_time = 120

    step = 30
    linewidth = 3
    dotsize=70

    # mystique_file = ["resnet18-scenario_1-mystique-corrects_per_10_sec.json", ""]
    # static_file = ["resnet18-scenario_1-static-corrects_per_10_sec.json", ""]
    # mm_file = ["corrects-resnet18_lr0.001-scenario_0-type_1-clear-resnet18-wide_resnet50_2.json", ""]
    # mystique_file = ["", ""]
    # static_file = ["", ""]
    # mm_file = ["", ""]
    # orin_file = ["student_file-scenario_0-type_1-clear-resnet34-wide_resnet101_2.json", ""]

    mystique_file = [
        resnet34_files["dc-st"]["correct"],
        ""
    ]
    static_file = [
        resnet34_files["dc-s"]["correct"],
        ""
    ]
    mm_file = [
        resnet34_files["eomu"]["correct"],
        ""
    ]
    orin_file = [
        resnet34_files["ekya"]["correct"],
        ""
    ]

    x_ticks, y_ticks = 120, 30
    start_y, end_y = 40, 100
    width, height, dpi = 800, 80, 100
    fontsize = 15

    # save_file = "accuracy-over-time-resnet34-overall.pdf"
    save_file = "figure_10_b.pdf"

    start_window = 0
    window_cnt = 9
    unit_sec = 15
    shift_left = 60
    option = 1

    start_range = start_window * window_time * fps + shift_left * fps
    end_range = start_range + window_cnt * window_time * fps

    draw([90, 390, 690, 990], [])

    # draw with end points
    mystique_file = [
        resnet34_files["dc-st"]["correct"],
        resnet34_files["dc-st"]["end_point"]
    ]
    static_file = [
        resnet34_files["dc-s"]["correct"],
        resnet34_files["dc-s"]["end_point"]
    ]
    mm_file = [
        resnet34_files["eomu"]["correct"],
        resnet34_files["eomu"]["end_point"]
    ]
    orin_file = [
        resnet34_files["ekya"]["correct"],
        resnet34_files["ekya"]["end_point"]
    ]

    x_ticks, y_ticks = 120, 10
    start_y, end_y = 40, 100
    width, height, dpi = 350, 200, 100
    fontsize = 15

    # save_file = "accuracy-over-time-resnet34-part1.pdf"
    save_file = "figure_10_e.pdf"

    start_window = 1
    window_cnt = 2
    unit_sec = 15
    shift_left = 0
    option = 2

    start_range = start_window * window_time * fps + shift_left * fps
    end_range = start_range + window_cnt * window_time * fps

    draw([], [180, ])

    x_ticks, y_ticks = 120, 10
    start_y, end_y = 40, 100
    width, height, dpi = 350, 200, 100
    fontsize = 15

    # save_file = "accuracy-over-time-resnet34-part2.pdf"
    save_file = "figure_10_f.pdf"

    start_window = 6
    window_cnt = 2
    unit_sec = 15
    shift_left = 0
    option = 2

    start_range = start_window * window_time * fps + shift_left * fps
    end_range = start_range + window_cnt * window_time * fps

    draw([], [720, 840 ])