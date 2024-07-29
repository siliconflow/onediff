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
        --exp-name baseline-lora-lora_multiple \
        --output-images

    # Baseline for ComfyUI_IPAdapter_plus
    python3 scripts/text_to_image.py \
        --comfy-port $COMFY_PORT \
        -w $WORKFLOW_BASIC/ComfyUI_IPAdapter_plus/ipadapter_advanced.json \
        --exp-name baseline-ipadapter_advanced \
        --output-images
else
    echo "CI is equal to 1"
fi

# Speedup for lora and lora_multiple with baseline directory
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_DIR/lora_speedup.json $WORKFLOW_DIR/lora_multiple_speedup.json \
    --baseline-dir $STANDARD_OUTPUT/test_lora_speedup \
    --ssim-threshold 0.6

# Speedup for ComfyUI_IPAdapter_plus with baseline directory
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
