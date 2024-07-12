#!/bin/bash
# set -x
set -e

docker pull oneflowinc/onediff-benchmark-pro-default:2024.01.18-cu121
if [ $? != 0 ]; then
  echo "failed to pull docker image"
fi

sh download_models.sh models
if [ $? != 0 ]; then
  echo "failed to download models"
fi

docker run -it --rm --gpus all --shm-size 12g --ipc=host --security-opt seccomp=unconfined --privileged=true \
  -v `pwd`:/benchmark \
  oneflowinc/onediff-benchmark-pro-default:2024.01.18-cu121 \
  sh -c "cd /benchmark && sh run_all_benchmarks.sh -m models -o benchmark.md"
