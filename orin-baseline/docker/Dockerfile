FROM yoonsungkim/dacapo-artifacts-orin:latest

WORKDIR /dacapo.artifacts.orin
COPY . /dacapo.artifacts.orin

ENV PROJECT_HOME="/dacapo.artifacts.orin"
ENV DATA_HOME="/data"
ENV OUTPUT_ROOT="/data"
ENV RAY_DEDUP_LOGS=0

ENTRYPOINT ["/bin/bash"]