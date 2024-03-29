import os
import json
import ray
import csv
import shutil
import shlex
import argparse
import subprocess
import numpy as np
from pathlib import Path


parser = argparse.ArgumentParser(description="Continuous learning experiments with FP on server-level GPU with")
parser.add_argument("--seed", type=int, default=128, help="seed")
parser.add_argument("--model", type=str, nargs="+", default=["resnet18"], help="list of model names (resnet18, resnet34, vit_b_32)")


PROJECT_HOME = Path(os.environ["PROJECT_HOME"])
DATA_HOME = Path(os.environ["DATA_HOME"])
OUTPUT_ROOT = Path(os.environ["OUTPUT_ROOT"])

STUDENT_WEIGHTS = {
    "resnet18": DATA_HOME / "weight/resnet18.pth",
    "resnet34": DATA_HOME / "weight/resnet34.pth",
    "vit_b_32": DATA_HOME / "weight/vit_b_32.pth",
}

MAX_EPOCH = 30
NUM_GPU = int(os.environ["NUM_GPU"])

ray.init(num_gpus=4,
         num_cpus=48)


@ray.remote(num_gpus=1)
def run_on_single_gpu(model: str,
                      seed: int,
                      cl_type: str,
                      weight_path: str,
                      output_root: Path,
                      scenario_path: str):
    log_path = Path(output_root / "log")
    log_path.mkdir(parents=True, exist_ok=True)

    scenario_name = os.path.basename(scenario_path).split(".")[0]
    output_path = Path(output_root / "output" / f"{scenario_name}")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[experiment info] model: {model}, "
          f"cl type: {cl_type}, "
          f"scenario: {scenario_name}")

    script_path = PROJECT_HOME / "fp-cl" / "run_fp_cl.py"

    cmd = f"python {str(script_path)} " \
          f"--scenario-path {scenario_path} " \
          f"--student {model} " \
          f"--weight-path {weight_path} " \
          f"--output-root {str(output_path)} " \
          f"--max-epoch {MAX_EPOCH} " \
          f"--seed {seed}"
    
    log_name = f"{model}-{scenario_name}"
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

    SCNEARIO_PATH = DATA_HOME / "dataset/bdd100k/6-scenarios"
    scenario_paths = sorted(os.listdir(SCNEARIO_PATH))

    tasks = []

    job_id = 0
    for model in model_list:
        cl_name = "fp"
        output_root = OUTPUT_ROOT / "output" / cl_name / model

        weight_path = STUDENT_WEIGHTS[model]

        for scenario_path in scenario_paths:
            task = run_on_single_gpu.remote(model=model,
                                            seed=seed,
                                            cl_type=cl_name,
                                            weight_path=str(weight_path),
                                            output_root=output_root,
                                            scenario_path=str(Path(SCNEARIO_PATH) / scenario_path))
            tasks.append(task)
            job_id += 1

    for task in tasks:
        ray.get(task)