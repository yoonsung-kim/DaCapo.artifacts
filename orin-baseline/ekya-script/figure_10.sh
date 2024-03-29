PWD_PATH="${PWD}"
DATASET_PATH="${DATA_HOME}/dataset/bdd100k/6-scenarios"
WEIGHT_PATH="${DATA_HOME}/weight"
PROFILE_PATH="${PWD}/ekya_orin/profile"
LOG_PATH="${OUTPUT_ROOT}/output/orin_high-ekya"

mkdir -p ${LOG_PATH}

python ${PROJECT_HOME}/ekya_orin/accuracy_over_time/run.py    --mode default                  \
                                                              --student resnet18              \
                                                              --teacher wide_resnet50_2       \
                                                              --scene s1                      \
                                                              --pwd_path ${PWD_PATH}          \
                                                              --dataset_path ${DATASET_PATH}  \
                                                              --weight_path ${WEIGHT_PATH}    \
                                                              --profile_path ${PROFILE_PATH}  \
                                                              --log_path ${LOG_PATH}

python ${PROJECT_HOME}/ekya_orin/accuracy_over_time/run.py    --mode default                  \
                                                              --student resnet34              \
                                                              --teacher wide_resnet101_2      \
                                                              --scene s1                      \
                                                              --pwd_path ${PWD_PATH}          \
                                                              --dataset_path ${DATASET_PATH}  \
                                                              --weight_path ${WEIGHT_PATH}    \
                                                              --profile_path ${PROFILE_PATH}  \
                                                              --log_path ${LOG_PATH}