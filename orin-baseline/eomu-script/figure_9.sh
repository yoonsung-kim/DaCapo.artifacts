#!/bin/bash

# Define the list of models
models=("resnet18" "resnet34" "vit_b_32")

# For the 6 scenarios
echo ">>>>> Start running DaCapo-EOMU with 6 scenarios >>>>>"
python "${PROJECT_HOME}/eomu/run_eomu_experiments.py" --model "${models[@]}"
echo "<<<<< End of running DaCapo-EOMU with 6 scenarios <<<<<"

