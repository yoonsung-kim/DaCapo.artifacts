#!/bin/bash

docker pull dustynv/pytorch:1.11-r35.3.1
docker build -t orin .
docker run  -dit                    \
            --runtime nvidia        \
            --name orin             \
            -v /mnt/hdd/data/:/mnt  \
            --ipc=host              \
            orin:latest
docker exec -it orin /bin/bash