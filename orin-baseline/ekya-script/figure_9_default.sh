MODE="default"
PWD_PATH="${PWD}"
DATASET_PATH="${DATA_HOME}/dataset/bdd100k/6-scenarios"
WEIGHT_PATH="${DATA_HOME}/weight"
PROFILE_PATH="${PWD}/ekya_orin/profile"
LOG_PATH="${OUTPUT_ROOT}/output/orin_high-ekya"

mkdir -p ${LOG_PATH}

python ${PROJECT_HOME}/ekya_orin/accuracy_over_window/run.py  --mode ${MODE}                  \
                                                              --pwd_path ${PWD_PATH}          \
                                                              --dataset_path ${DATASET_PATH}  \
                                                              --weight_path ${WEIGHT_PATH}    \
                                                              --profile_path ${PROFILE_PATH}  \
                                                              --log_path ${LOG_PATH}