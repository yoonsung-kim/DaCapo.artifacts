import csv
import json
import argparse
import itertools
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


parser = argparse.ArgumentParser(description="summarize results for figure. 9")
parser.add_argument("--output-root", type=str, help="root path of output")
parser.add_argument("--summary-root", type=str, default="./", help="root path of summarized results")


# ╭────────────╮
# │ PARAMETERS │
# ╰────────────╯
# SCENARIO = "scenario_0-type_1-clear"
SCENARIO = "s1"
RANGE_SEC = (160 + 480, 330 + 480)
MODELS = [
    "resnet18",
    "vit_b_32",
    "resnet34",
]


if __name__ == "__main__":
    args = parser.parse_args()
    output_root = Path(args.output_root)
    summary_root = Path(args.summary_root)
    summary_dir = summary_root / "figure_11"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # ╭──────╮
    # │ DATA │
    # ╰──────╯
    data = {}
    for model in MODELS:
        data[model] = {}
        for mode in ["static", "dynamic"]:
            data[model][mode] = {
                "decision": [],
                "acc": 0,
            }
    dynamic_start_img_idx = RANGE_SEC[0] * 30
    dynamic_end_img_idx = RANGE_SEC[1] * 30
    num_of_imgs = (RANGE_SEC[1] - RANGE_SEC[0]) * 30


    def push(lst: List[Tuple[str, float]], phase: str, value: int):
        sec = float(value) / 30
        if lst != [] and lst[-1][0] == phase:
            lst[-1] = (phase, lst[-1][1] + sec)
        else:
            lst.append((phase, sec))


    # static
    for model in MODELS:
        # m = [x in model.name for x in MODELS]
        # if any(m):
        #     model_name = MODELS[m.index(True)]
        # else:
        #     continue

        data_path = output_root / "spatial" /  model / "output" / SCENARIO
        log_path = data_path / "result.csv" # next(data_path.glob("static*csv"))
        acc_path = data_path / "result.json" # next(data_path.glob("static*json"))

        log_data = pd.read_csv(log_path)
        with open(acc_path, "r") as f:
            acc_data = json.load(f)

        acc_data = list(itertools.chain(*acc_data.values()))

        acc_img = 0
        state = 0
        correct = 0
        for _, row in log_data.iterrows():
            num_of_phase1_imgs = int(row["phase 1 # of imgs"])
            num_of_phase2_imgs = int(row["phase 2 # of imgs"])

            for phase, num_of_images in [
                ("train", num_of_phase1_imgs),
                ("label", num_of_phase2_imgs),
            ]:
                acc_img += num_of_images

                if state == 0 and acc_img > dynamic_start_img_idx:
                    push(
                        data[model]["static"]["decision"],
                        phase,
                        acc_img - dynamic_start_img_idx,
                    )
                    correct += sum(acc_data[dynamic_start_img_idx:acc_img])
                    state = 1
                elif state == 1 and acc_img <= dynamic_end_img_idx:
                    push(data[model]["static"]["decision"], phase, num_of_images)
                    correct += sum(acc_data[acc_img - num_of_images : acc_img])
                elif state == 1:
                    push(
                        data[model]["static"]["decision"],
                        phase,
                        dynamic_end_img_idx - acc_img + num_of_images,
                    )
                    correct += sum(acc_data[acc_img - num_of_images : dynamic_end_img_idx])
                    break
            else:
                continue

            break
        data[model]["static"]["acc"] = float(correct) / 36 / num_of_imgs * 3600

    # dynamic
    for model in MODELS:
        # m = [x in model.name for x in MODELS]
        # if any(m):
        #     model_name = MODELS[m.index(True)]
        # else:
        #     continue

        # data_path = output_root / "spatial" /  model / SCENARIO
        # log_path = data_path / "result.csv" # next(data_path.glob("static*csv"))
        # acc_path = data_path / "result.json" # next(data_path.glob("static*json"))

        data_path = output_root / "spatiotemporal" /  model / "output" / SCENARIO
        log_data = pd.read_csv(data_path / "phase_log.csv")
        with open(data_path / "result.json", "r") as f:
            acc_data = json.load(f)
        acc_data = list(itertools.chain(*acc_data))

        acc_img = 0
        state = 0
        correct = 0
        for _, row in log_data.iterrows():
            num_of_images = row["# of images"]
            assert isinstance(num_of_images, int)

            phase = row["Phase name"]
            phase = "train" if "train" in phase else "label"

            acc_img += num_of_images

            if state == 0 and acc_img > dynamic_start_img_idx:
                push(
                    data[model]["dynamic"]["decision"],
                    phase,
                    acc_img - dynamic_start_img_idx,
                )
                correct += sum(acc_data[dynamic_start_img_idx:acc_img])
                state = 1
            elif state == 1 and acc_img <= dynamic_end_img_idx:
                push(data[model]["dynamic"]["decision"], phase, num_of_images)
                correct += sum(acc_data[acc_img - num_of_images : acc_img])
            elif state == 1:
                push(
                    data[model]["dynamic"]["decision"],
                    phase,
                    dynamic_end_img_idx - acc_img + num_of_images,
                )
                correct += sum(acc_data[acc_img - num_of_images : dynamic_end_img_idx])
                break
        data[model]["dynamic"]["acc"] = float(correct) / 36 / num_of_imgs * 3600

    fig, axes = plt.subplots(len(MODELS) * 2, 1, figsize=(5.0, 3.0))
    axes: List[Axes]

    ax_num = 0
    for model in MODELS:
        for mode in ["static", "dynamic"]:
            left = 0
            for phase, value in data[model][mode]["decision"]:
                ax = axes[ax_num]
                ax.barh(
                    "foo",
                    value,
                    left=left,
                    color="#94bcbf" if phase == "train" else "#b8d9b0",
                    edgecolor="black",
                )
                ax.set_xlim(-1, 121 * num_of_imgs / 3600)
                # ax.margins(y=1.5)
                ax.set_axis_off()
                left += value

            num_of_trains = sum(
                value for phase, value in data[model][mode]["decision"] if phase == "train"
            )
            train_ratio = round(float(num_of_trains) * 100 / 120 / num_of_imgs * 3600)
            print(f"[{model}-{mode}]")
            print("acc: ", data[model][mode]["acc"])
            print(f"ratio: {train_ratio}:{100 - train_ratio}")
            print()
            ax_num += 1

    fig.tight_layout()
    # plt.show()
    plt.savefig(str(summary_dir / "figure_11.pdf"))

    f = open(summary_dir / "figure_11-sheet.csv", "w")
    csv_writer = csv.writer(f)

    csv_writer.writerow([
        "Model",
        "Name",
        "Phase ratio",
        "Acc. Improv."
    ])

    prefix_names = ["(a)", "(b)", "(c)"]
    for i in range(len(MODELS)):
        model = MODELS[i]
        prefix_name = prefix_names[i]
    
        dc_s_acc = data[model]["static"]["acc"]
        num_of_trains = sum(value for phase, value in data[model]["static"]["decision"] if phase == "train")
        dc_s_train_ratio = round(float(num_of_trains) * 100 / 120 / num_of_imgs * 3600)
        dc_s_label_ratio = 100 - dc_s_train_ratio
        
        dc_st_acc = data[model]["dynamic"]["acc"]
        num_of_trains = sum(value for phase, value in data[model]["dynamic"]["decision"] if phase == "train")
        dc_st_train_ratio = round(float(num_of_trains) * 100 / 120 / num_of_imgs * 3600)
        dc_st_label_ratio = 100 - dc_st_train_ratio

        dc_s_ratio = f"{dc_s_train_ratio:02d}:{dc_s_label_ratio:02d}"
        dc_st_ratio = f"{dc_st_train_ratio:02d}:{dc_st_label_ratio:02d}"
        acc_improv = f"{dc_st_acc - dc_s_acc:1.1f}%"
        
        csv_writer.writerow([
            f"{prefix_name} {model}",
            "DC-S",
            dc_s_ratio,
            "-"
        ])

        csv_writer.writerow([
            f"{prefix_name} {model}",
            "DC-ST",
            dc_st_ratio,
            acc_improv
        ])

