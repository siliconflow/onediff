#!/bin/bash
set -e

STANDARD_OUTPUT=/share_nfs/hf_models/comfyui_resources/standard_output
COMFY_PORT=8188
WORKFLOW_BASIC=resources/workflows/baseline
WORKFLOW_DIR=resources/workflows/oneflow

# Run sdxl-control-lora-speedup workflow
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_DIR/sdxl-control-lora-speedup.json

if [ "$CI" != "1" ]; then
    echo "CI is not equal to 1 set STANDARD_OUTPUT=results"
    STANDARD_OUTPUT=results
    # Baseline for lora and lora_multiple
    python3 scripts/text_to_image.py \
        -w $WORKFLOW_BASIC/lora.json $WORKFLOW_BASIC/lora_multiple.json \
        --output-dir $STANDARD_OUTPUT \
        --exp-name baseline-lora-lora_multiple \
        --output-images

    # Baseline for ComfyUI_IPAdapter_plus
    python3 scripts/text_to_image.py \
        --comfy-port $COMFY_PORT \
        -w $WORKFLOW_BASIC/ComfyUI_IPAdapter_plus/ipadapter_advanced.json \
        --output-dir $STANDARD_OUTPUT \
        --exp-name baseline-ipadapter_advanced \
        --output-images
else
    echo "CI is equal to 1"
fi

# Speedup for lora and lora_multiple with baseline directory
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_DIR/lora_speedup.json $WORKFLOW_DIR/lora_multiple_speedup.json \
    --output-dir $STANDARD_OUTPUT \
    --exp-name lora-lora_multiple-speedup \
    --baseline-dir $STANDARD_OUTPUT/baseline-lora-lora_multiple \
    --output-images

# Speedup for ComfyUI_IPAdapter_plus with baseline directory
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_DIR/ComfyUI_IPAdapter_plus/ipadapter_advanced.json \
    --output-dir $STANDARD_OUTPUT \
    --exp-name ipadapter_advanced-speedup \
    --baseline-dir $STANDARD_OUTPUT/baseline-ipadapter_advanced \
    --ssim-threshold 0.8 \
    --output-images
