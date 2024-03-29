import os
import csv
import argparse
from pathlib import Path
from scipy.stats import gmean


parser = argparse.ArgumentParser(description="summarize results for table v")
parser.add_argument("--output-root", type=str, help="root path of output")
parser.add_argument("--summary-root", type=str, default="./", help="root path of summarized results")


NAMES = [
    "spatial-active_cl",
    "spatiotemporal-active_cl",
    "spatial",
    "spatiotemporal",
]

MODELS = [
    "resnet18",
    "vit_b_32",
    "resnet34",
]

CSV_RAW_DATA = {}

NUM_SCENARIO = 6
NUM_WARM_UP = 4


if __name__ == "__main__":
    args = parser.parse_args()
    output_root = Path(args.output_root)
    summary_root = Path(args.summary_root)

    for model in MODELS:
        CSV_RAW_DATA[model] = {}

        for name in NAMES:
            CSV_RAW_DATA[model][name] = {}

            gmean_accs = []

            for s in range(1, NUM_SCENARIO+1):
                # print(s, end=" ")
                s_name = f"s{s}"
                # print(NAME_DICT[name], end=" ")
                # TODO: make it consistent to every baselines
                output_path = output_root / name / model / "output" / s_name / "result.csv"

                gmean_acc = 0.
                if os.path.isfile(output_path):
                    with open(output_path) as f:
                        row_cnt = 0
                        csv_reader = csv.reader(f)
                        for row in csv_reader:
                            row_cnt += 1
                            
                    if row_cnt == 15:
                        accs = []
                        with open(output_path) as f:
                            csv_reader = csv.reader(f)

                            next(csv_reader)
                            
                            for row in csv_reader:
                                window_idx, acc = int(row[0]), float(row[1])
                                accs.append(acc)
                        
                        accs = accs[NUM_WARM_UP:]
                        gmean_acc = gmean(accs)
                        # print(gmean_acc)
                        gmean_accs.append(gmean_acc)
                
                # CSV_RAW_DATA[model][name][s_name] = gmean_acc
            CSV_RAW_DATA[model][name] = gmean(gmean_accs)
            print(f"{model} {name} acc: {CSV_RAW_DATA[model][name]}")

    summary_dir = Path(summary_root / "table_v")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / "table_v-sheet.csv"
    f = open(summary_file, "w")
    csv_writer = csv.writer(f)

    header = ["Model", "Sampling method", "DC-S", "DC-ST"]
    csv_writer.writerow(header)

    for model in MODELS:
        dc_s_uni = CSV_RAW_DATA[model]["spatial"]
        dc_s_act = CSV_RAW_DATA[model]["spatial-active_cl"]

        dc_st_uni = CSV_RAW_DATA[model]["spatiotemporal"]
        dc_st_act = CSV_RAW_DATA[model]["spatiotemporal-active_cl"]
        
        row = [
            model,
            "Uniform",
            f"{dc_s_uni:2.1f}%",
            f"{dc_st_uni:2.1f}%",
        ]
        csv_writer.writerow(row)

        row = [
            model,
            "Active-CL",
            f"{dc_s_act:2.1f}% ({dc_s_act-dc_s_uni:+1.1f}%)",
            f"{dc_st_act:2.1f}% ({dc_st_act-dc_st_uni:+1.1f}%)",
        ]
        csv_writer.writerow(row)