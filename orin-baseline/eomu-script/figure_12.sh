#!/bin/bash

# Define the list of models
models=("resnet18")

# For the extreme scenario
echo ">>>>> Start running DaCapo-EOMU with 2 extreme scenarios >>>>>"
python "${PROJECT_HOME}/eomu/run_eomu_experiments.py" --model "${models[@]}" --extreme-scenario
echo "<<<<< End of running DaCapo-EOMU with 2 extreme scenarios <<<<<"