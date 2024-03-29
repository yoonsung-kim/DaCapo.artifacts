#!/bin/bash

# Define the list of models
models=("resnet18")

# Define the list of cl-types
cl_types=("spatial" "spatiotemporal" "ekya" "spatial-active_cl" "spatiotemporal-active_cl")

# Loop over each cl-type
for cl_type in "${cl_types[@]}"; do
    # Print the start message
    echo ">>>>> Start running $cl_type >>>>>"
    
    # Run the experiments for each model
    python "${PROJECT_HOME}/script/run_dacapo_experiments.py" --cl-type $cl_type --model "${models[@]}"
    
    # Print the end message
    echo "<<<<< End of running $cl_type <<<<<"
done

# For the extreme scenario
echo ">>>>> Start running spatiotemporal with extreme scenario >>>>>"
python "${PROJECT_HOME}/script/run_dacapo_experiments.py" --cl-type spatiotemporal --model "${models[@]}" --extreme-scenario
echo "<<<<< End of running spatiotemporal with extreme scenario <<<<<"