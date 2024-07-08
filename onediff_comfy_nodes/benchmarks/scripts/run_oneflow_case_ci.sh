#!/bin/bash
set -e

STANDARD_OUTPUT=/share_nfs/hf_models/comfyui_resources/standard_output
COMFY_PORT=8188
WORKFLOW_DIR=resources/workflows/oneflow

python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_DIR/sdxl-control-lora-speedup.json

# # Baseline
# python3 scripts/text_to_image.py \
#     -w resources/baseline/lora.json resources/baseline/lora_multiple.json \
#     --output-images
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_DIR/lora_speedup.json $WORKFLOW_DIR/lora_multiple_speedup.json \
    --baseline-dir $STANDARD_OUTPUT/test_lora_speedup \
    --ssim-threshold 0.6

# # Baseline
# python3 scripts/text_to_image.py \
#      --comfy-port $COMFY_PORT \
#     -w resources/baseline/ComfyUI_IPAdapter_plus/ipadapter_advanced.json \
#     --output-images
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_DIR/ComfyUI_IPAdapter_plus/ipadapter_advanced.json \
    --baseline-dir $STANDARD_OUTPUT/test_ipa
# --output-images \

python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_DIR/txt2img.json \
    --ssim-threshold 0.6 \
    --baseline-dir $STANDARD_OUTPUT/txt2img/imgs # --output-images
