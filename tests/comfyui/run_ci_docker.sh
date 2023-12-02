#!/bin/bash

# export COMFYUI_PATH=/share_nfs/comfyui-ci-assets/ComfyUI
# export COMFYUI_SPEEDUP_PATH=/share_nfs/comfyui-ci-assets/ComfyUI/custom_nodes/comfyui-speedup
# export SELENIUM_IMAGE=registry.cn-beijing.aliyuncs.com/oneflow/standalone-chrome:119.0-chromedriver-119.0-grid-4.15.0-20231129
# export SELENIUM_CONTAINER_NAME=selenium-container
# export COMFYUI_PORT=8855
# export ONEIDFF_CONTAINER_NAME=onediff-test
# export ONEDIFF_IMAGE=registry.cn-beijing.aliyuncs.com/oneflow/onediff-pro:cu121
# export SDXL_BASE=/share_nfs/hf_models/sd_xl_base_1.0.safetensors
# export UNET_INT8=/share_nfs/hf_models/unet_int8

docker rm -f $SELENIUM_CONTAINER_NAME || true
docker pull $SELENIUM_IMAGE
docker run --network host \
    --shm-size="2g" --rm --name $SELENIUM_CONTAINER_NAME \
    -d $SELENIUM_IMAGE sleep 5400
docker exec -d $SELENIUM_CONTAINER_NAME /opt/bin/entry_point.sh

docker rm -f $ONEIDFF_CONTAINER_NAME || true
docker pull $ONEDIFF_IMAGE
docker run --rm --gpus=all -d --privileged --shm-size=8g \
    --network host --pids-limit 2000 \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v $COMFYUI_PATH:/app/ComfyUI \
    -v $COMFYUI_SPEEDUP_PATH:/app/ComfyUI/custom_nodes/comfyui-speedup \
    -v $SDXL_BASE:/app/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors:ro \
    -v $UNET_INT8:/app/ComfyUI/models/unet_int8/unet_int8:ro \
    --name $ONEIDFF_CONTAINER_NAME \
    $ONEDIFF_IMAGE \
    sleep 5400

docker exec -d $ONEIDFF_CONTAINER_NAME python3 /app/ComfyUI/main.py --port $COMFYUI_PORT