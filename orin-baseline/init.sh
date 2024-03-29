export EKYA_HOME="${PWD}/ext/ekya"
export PYTHONPATH="${PYTHONPATH}:${EKYA_HOME}"
export PYTHON3PATH="${PYTHON3PATH}:${EKYA_HOME}"
export RAY_DEDUP_LOGS=0

python init.py

rm ./ekya_orin/profile/src
ln -s ${PWD}/ekya_common ./ekya_orin/profile/src

rm ./ekya_orin/accuracy_over_time/data/profile
mkdir ./ekya_orin/accuracy_over_time/data
ln -s ${PWD}/ekya_orin/profile ./ekya_orin/accuracy_over_time/data/profile

rm ./ekya_orin/accuracy_over_time/src/src
ln -s ${PWD}/ekya_common ./ekya_orin/accuracy_over_time/src/src

rm ./ekya_orin/accuracy_over_window/data/profile
mkdir ./ekya_orin/accuracy_over_window/data
ln -s ${PWD}/ekya_orin/profile ./ekya_orin/accuracy_over_window/data/profile

rm ./ekya_orin/accuracy_over_window/src/src
ln -s ${PWD}/ekya_common ./ekya_orin/accuracy_over_window/src/src

rm ./ekya_orin/motivation/data/profile
mkdir ./ekya_orin/motivation/data
ln -s ${PWD}/ekya_orin/profile ./ekya_orin/motivation/data/profile

rm ./ekya_orin/motivation/src/src
ln -s ${PWD}/ekya_common ./ekya_orin/motivation/src/src