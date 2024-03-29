import os
import json
import shlex
import shutil
import argparse
import subprocess
from pathlib import Path


parser = argparse.ArgumentParser(description="dacapo experiment launcher")
parser.add_argument("--seed", type=int, default=128, help="seed")
parser.add_argument("--model", type=str, nargs="+", default=["resnet18"], help="list of model names (resnet18, resnet34, vit_b_32)")
parser.add_argument("--extreme-scenario", action="store_true", help="run experiment with extreme scenario dataset")


PROJECT_HOME = Path(os.environ["PROJECT_HOME"]) / "eomu"
DATA_HOME = Path(os.environ["DATA_HOME"])
OUTPUT_ROOT = Path(os.environ["OUTPUT_ROOT"])

STUDENT_WEIGHTS = {
    "resnet18": DATA_HOME / "weight/resnet18.pth",
    "resnet34": DATA_HOME / "weight/resnet34.pth",
    "vit_b_32": DATA_HOME / "weight/vit_b_32.pth",
}

CONFIG_TEMPLATE_PATHS = {
    "resnet18": PROJECT_HOME / "config/resnet18-wide_resnet50_2.json",
    "resnet34": PROJECT_HOME / "config/resnet34-wide_resnet101_2.json",
    "vit_b_32": PROJECT_HOME / "config/vit_b_32-vit_b_16.json",
}

ITER_TIMES = {
    "resnet18": {
        "train_iter_time": 0.03050964569,
        "label_iter_time": 0.01704849357
    },
    "resnet34": {
        "train_iter_time": 0.04500128008,
        "label_iter_time": 0.03306519073
    },
    "vit_b_32": {
        "train_iter_time": 0.0630437546,
        "label_iter_time": 0.01521971349
    }
}


def run(config_template_path: str,
        train_iter_time: float,
        label_iter_time: float,
        student_weight: str,
        output_root: Path,
        # output_dir: str,
        scenario_path: str):
    log_path = Path(output_root / "log")
    log_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(output_root / "config")
    config_path.mkdir(parents=True, exist_ok=True)

    scenario_name = os.path.basename(scenario_path).split(".")[0]
    output_path = Path(output_root / "output" / f"{scenario_name}")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[experiment info] model: {model}, "
          f"cl type: eomu, "
          f"scenario: {scenario_name}")

    with open(config_template_path) as f:
        template_config = json.load(f)
    template_config["num_classes"] = 9
    template_config["window_time"] = 10
    template_config["train_iter_time"] = train_iter_time
    template_config["label_iter_time"] = label_iter_time
    template_config["num_workers"] = 12
    template_config["student_weight"] = student_weight
    template_config["output_root"] = str(output_path)
    template_config["scenario_path"] = scenario_path

    log_name = f"{model}-{scenario_name}"

    dst_config_path = str(config_path / f"{log_name}.json")
    with open(dst_config_path, "w") as f:
        json.dump(template_config, f, indent=4)

    cmd = f"python {str(PROJECT_HOME)}/run_eomu.py " \
            f"--config-path {dst_config_path}"
    
    out_path = log_path / f"{log_name}.stdout"
    err_path = log_path / f"{log_name}.stderr"

    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        handle = subprocess.Popen(
            shlex.split(cmd),
            env=dict(os.environ),
            stdout=out, stderr=err)
        handle.communicate()

if __name__ == "__main__":
    args = parser.parse_args()
    seed = args.seed
    model_list = args.model
    is_extreme_scenario = args.extreme_scenario

    if is_extreme_scenario:
        SCNEARIO_PATH = DATA_HOME / "dataset/bdd100k/resized-extreme-2-scenarios"
    else:
        SCNEARIO_PATH = DATA_HOME / "dataset/bdd100k/resized-6-scenarios"
    scenario_paths = sorted(os.listdir(SCNEARIO_PATH))

    for model in model_list:
        output_root = OUTPUT_ROOT / "output" / "eomu" / model

        for scenario_path in sorted(scenario_paths):
            student_weight_path = STUDENT_WEIGHTS[model]
            config_template_path = CONFIG_TEMPLATE_PATHS[model]
            train_iter_time = ITER_TIMES[model]["train_iter_time"]
            label_iter_time = ITER_TIMES[model]["label_iter_time"]

            run(config_template_path=config_template_path,
                train_iter_time=train_iter_time,
                label_iter_time=label_iter_time,
                student_weight=str(student_weight_path),
                output_root=output_root,
                scenario_path=str(SCNEARIO_PATH / scenario_path))
