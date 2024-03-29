#!/bin/bash

# Define the list of models
models=("resnet18" "resnet34" "vit_b_32")

# For the 6 scenarios
echo ">>>>> Start running RTX3090-CL with 6 scenarios >>>>>"
python "${PROJECT_HOME}/fp-cl/run_fp_cl_experiments.py" --model "${models[@]}"
echo "<<<<< End of running RTX3090-CL with 6 scenarios <<<<<"