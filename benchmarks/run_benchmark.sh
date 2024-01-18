#!/bin/bash
set -x

export ONEFLOW_RUN_GRAPH_BY_VM=1

if [ $# != 1 ]; then
  echo "Usage: bash run_benchmark.sh /path/model" && exit 1
fi
BENCHMARK_MODEL_PATH=$1

######################################################################################
echo "Run SD1.5(FP16) 512x512..."
python3 ./text_to_image.py --model ${BENCHMARK_MODEL_PATH}/stable-diffusion-v1-5 --warmup 5 --height 512 --width 512

######################################################################################
echo "Run SD2.1(FP16) 768x768..."
python3 ./text_to_image.py --model ${BENCHMARK_MODEL_PATH}/stable-diffusion-2-1 --warmup 5 --height 768 --width 768

######################################################################################
echo "Run SDXL(FP16) 1024x1024..."
python3 ./text_to_image_sdxl_fp16.py --model ${BENCHMARK_MODEL_PATH}/stable-diffusion-xl-base-1.0 --warmup 5 --height 1024 --width 1024

######################################################################################
echo "Run SDXL(INT8) 1024x1024..."
python3 ./text_to_image_sdxl_quant.py --model ${BENCHMARK_MODEL_PATH}/stable-diffusion-xl-base-1.0-int8 --warmup 5 --height 1024 --width 1024
