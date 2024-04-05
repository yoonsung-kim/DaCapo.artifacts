# DaCapo: Accelerating Continuous Learning in Autonomous Systems for Video Analytics
* [0. Clone GitHub repository](#0-clone-github-repository)
* [1. Installation](#1-installation)
  + [1.1. Setup Docker image](#11-setup-docker-image)
  + [1.2. Download data](#12-download-data)
  + [1.3. Generate Docker container](#13-generate-docker-container)
* [2. Run experiment](#2-run-experiment)
  + [2.1. Experiment with NVIDIA RTX 3090](#21-experiment-with-nvidia-rtx-3090)
    - [2.1.1. Generate Docker container](#211-generate-docker-container)
    - [2.1.2. Run script for DaCapo systems](#212-run-script-for-dacapo-systems)
    - [2.1.3. Run script for RTX3090-CL baseline](#213-run-script-for-rtx3090-cl-baseline)
  + [2.2. Experiment with NVIDIA Jetson Orin](#22-experiment-with-nvidia-jetson-orin)
    - [2.2.1. Generate Docker container](#221-generate-docker-container)
    - [2.2.2. Run script on default power settings](#222-run-script-on-default-power-settings)
    - [2.2.3. Run script on 30W power settings](#223-run-script-on-30w-power-settings)
* [3. Summarize experiment result](#3-summarize-experiment-result)
  + [3.1. Merge output directory](#31-merge-output-directory)
  + [3.2. Run script](#32-run-script)
  + [3.3. Expected summarized result](#33-expected-summarized-result)

#### Hardware platforms and continuous learning methods
1. NVIDIA RTX 3090 (24GB)
  - DaCapo systems
    - DaCapo-Spatial
    - DaCapo-Spatiotemporal
    - DaCapo-Ekya
  - RTX3090-CL
2. NVIDIA Jetson Orin (64GB)
  - Eyka
    - OrinLow-Ekya
    - OrinHigh-Ekya
  - EOMU
    - OrinHigh-Ekya

#### Tested environment

1. DaCapo systems and RTX3090 baseline
    - Host
      - Docker 24.0
    - Docker image
      - Ubuntu 18.04
    - GPU
      - NVIDIA RTX 3090 (24GB) *multiple GPUs can run experiments in parallel

2. Orin baselines: Ekya and EOMU
    - Host
      - Docker 20.10
    - Docker image
      - Ubuntu 20.04
    - GPU
      - NVIDIA Jetson Orin (64GB)

## 0. Clone GitHub repository

```shell
git clone --recursive https://github.com/yoonsung-kim/DaCapo.artifacts.git
cd DaCapo.artifacts
```

## 1. Installation

### 1.1. Setup Docker image

Pull base Docker images.

```shell
# for the system with NVIDIA RTX 3090
docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# for the system with NVIDIA Jetson Orin
docker pull yoonsungkim/dacapo-artifacts-orin
```

Generate Docker images for the systems:

1. Build Docker image on the system with NVIDIA RTX 3090

We can set ```NUM_GPU``` environment variable in Dockerfile to make the system run experiments in parallel.

```shell
# at docker/Dockerfile
...
# set the number of GPU
NUM_GPU=<integer>
...
```

```shell
docker build --no-cache -t dacapo-emulation -f docker/Dockerfile .
```

2. Build Docker image on the system with NVIDIA Jetson Orin

```shell
cd orin-baseline
docker build --no-cache -t orin-baseline -f docker/Dockerfile .
```

### 1.2. Download data

1. Download data.tar (about 7.5GB). This data includes all scenario datasets and the weights of DNN models. The download links are below:
    - [Google Drive](https://drive.google.com/drive/folders/1rNTPJXrPlkestSTRoxXDQZA93hTOZxmy?usp=sharing)

2. Decompress ```data.tar```. The directory hierarchy is as below:

```shell
data/
├── dataset # all BDD100K scenario datasets
└── weight  # initial weights for benchmarks
```

### 1.3. Generate Docker container

1. Docker container for DaCapo systems and RTX3090-CL baseline

```shell
# Run script on the system with NVIDIA RTX 3090
docker run -it -v <path-to-data-directory>:/data --ipc=host --name dacapo-emulation --gpus all dacapo-emulation:latest
```

2. Docker container for Orin baselines

```shell
# Run script on the system with NVIDIA Jetson Orin
docker run -it -v <path-to-data-directory>:/data --ipc=host --name orin-baseline --runtime nvidia orin-baseline:latest
```

## 2. Run experiment

All experiments generate their results in ```$OUTPUT_ROOT``` directory defined in Dockerfiles. The path in a Docker container is ```/data```, and the system saves the results in the ```/data/output``` directory. Note that the ```/data``` is mounted to the host system (i.e., ```docker run ... -v <path-to-data-directory>:/data ...```).

```shell
# in Dockerfile
... 
ENV OUTPUT_ROOT="/data"
...
```

The output directories for the both systems have the same hierarchy as follows:

1. DaCapo systems and RTX3090-CL baseline
```shell
data/output/
├── dacapo-ekya
├── fp # i.e., RTX3090-CL
├── spatial
├── spatial-active_cl
├── spatiotemporal
└── spatiotemporal-active_cl
```

2. Orin baselines
```shell
data/output/
├── eomu
├── orin_high-ekya
└── orin_low-ekya
```

After running all experiments, we should combine the directories into a single ```output``` directory. Once this is done, we can summarize experiment results by executing post-processing scripts.

### 2.1. Experiment with NVIDIA RTX 3090

#### 2.1.1. Generate Docker container

```shell
docker run -it -v <path-to-data-directory>:/data --ipc=host --name dacapo-emulation --gpus all dacapo-emulation:latest
```

#### 2.1.2. Run script for DaCapo systems

```shell
./script/run_all_benchmarks.sh
```

#### 2.1.3. Run script for RTX3090-CL baseline

```shell
cd fp-cl
./figure_9.sh
```

### 2.2. Experiment with NVIDIA Jetson Orin

#### 2.2.1. Generate Docker container

```shell
docker run -it -v <path-to-data-directory>:/data --ipc=host --name orin-baseline --runtime nvidia orin-baseline:latest
```

The Orin GPU offers two distinct power modes: (1) default and (2) 30W settings. To conduct experiments, we need to configure the GPU's power mode accordingly. Note that the system requires rebooting after changing the power mode.

#### 2.2.2. Run script on default power settings

1. Set power mode as default and reboot

```shell
# host system
sudo /usr/sbin/nvpmodel -m 0
sudo reboot

# ... after reboot

# check if the system outputs "0"
sudo /usr/sbin/nvpmodel -q

# attach to Docker container
```

2. Run script

Make sure that we need to execute scripts at ```WORKDIR``` (i.e., ```/dacapo.artifacts.orin```), please do not execute the scripts at ```ekya-script``` or ```eomu-script```.

```shell
# ignore any errors related to the path
./init.sh

# Ekya baseline
./ekya-script/figure_9_default.sh
./ekya-script/figure_10.sh
./ekya-script/figure_12.sh

# EOMU baseline
./eomu-script/figure_9.sh
./eomu-script/figure_12.sh
```

#### 2.2.3. Run script on 30W power settings

1. Set power mode as 30W and reboot

```shell
# host system
sudo /usr/sbin/nvpmodel -m 2
sudo reboot

# ... after reboot

# check if the system outputs "2"
sudo /usr/sbin/nvpmodel -q

# attach to Docker container
```

2. Run script

```shell
# ignore any errors related to the path
./init.sh

# Ekya baseline
./ekya-script/figure_9_30W.sh
```

## 3. Summarize experiment result

## 3.1. Merge output directory

After running the experiments, we can summarize the results by executing the post-processing scripts. However, before doing so, we need to merge the output directories from the two different systems to summarize the results together. In cases where some results are missing, the scripts will not include them in the summary. The hierarchy of the gathered output directory is as follows:

```shell
data/output/                  # required to generate:
├── dacapo-ekya               # Figure. 9
├── fp                        # Figure. 9
├── spatial-active_cl         # Table V
├── spatiotemporal-active_cl  # Table V
├── spatial                   # Figure. 9, 10 / Table V
├── spatiotemporal            # Figure. 9, 10, 11, 12 / Table V
├── eomu                      # Figure. 9, 10, 11, 12
├── orin_high-ekya            # Figure. 9, 10, 12
└── orin_low-ekya             # Figure. 9
```

## 3.2. Run script

1. Set environment variable

```shell
export OUTPUT_DIR=<output-directory>
export SUMMARY_DIR=<directory-to-save-summarized-result>
```

2. Run script

```shell
cd script/summarize
python ./figure_9.py --output-root $OUTPUT_DIR --summary-root $SUMMARY_DIR
python ./figure_10.py --output-root $OUTPUT_DIR --summary-root $SUMMARY_DIR
python ./figure_11.py --output-root $OUTPUT_DIR --summary-root $SUMMARY_DIR
python ./figure_12.py --output-root $OUTPUT_DIR --summary-root $SUMMARY_DIR
python ./table_v.py --output-root $OUTPUT_DIR --summary-root $SUMMARY_DIR
```

## 3.3. Expected summarized result

These scripts generate figures in ```.pdf``` format and tables in ```.csv``` format in the ```$SUMMARY_DIR``` directory. The files include (1) averaged accuracy, (2) changes in accuracy over time, and (3) the endpoints of retraining phase during the continuous learning on DaCapo systems and other baselines. The directory is as follows:

```shell
$SUMMARY_DIR/
├── figure_9/
│   └── figure_9-sheet.csv
├── figure_10/
│   ├── figure_10_a.pdf
│   ├── figure_10_b.pdf
│   ├── figure_10_c.pdf
│   ├── figure_10_d.pdf
│   ├── figure_10_e.pdf
│   └── figure_10_f.pdf
├── figure_11/
│   ├── figure_11-sheet.csv
│   └── figure_11.pdf
├── figure_12/
│   ├── figure_12_b.pdf
│   ├── figure_12_d.pdf
│   └── figure_12-sheet.csv
└── table_v/
    └── table_v-sheet.csv
```

These results facilitate comparison with the reported numbers and graphs in the paper. In the case of Figure 9, we provide the ```.xlsx``` file located at ```reference/figure_9.xlsx``` to verify the accuracy numbers of a bar graph.