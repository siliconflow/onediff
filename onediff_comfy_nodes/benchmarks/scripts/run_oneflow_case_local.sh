#!/bin/bash
set -e

COMFY_PORT=8188
STANDARD_OUTPUT=/share_nfs/hf_models/comfyui_resources/standard_output
WORKFLOW_BASIC=resources/workflows/baseline
WORKFLOW_DIR=resources/workflows/oneflow

# Run PuLID_ComfyUI baseline workflow
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_BASIC/PuLID_ComfyUI/PuLID_4-Step_lightning.json \
    --output-dir results \
    --exp-name PuLID_4-Step_lightning_baseline \
    --output-images

python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_BASIC/PuLID_ComfyUI/PuLID_IPAdapter_style_transfer.json \
    --output-dir results \
    --exp-name PuLID_IPAdapter_style_transfer_baseline \
    --output-images

# Run PuLID_ComfyUI oneflow workflow with baseline directory
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_DIR/PuLID_ComfyUI/PuLID_4-Step_lightning.json \
    --output-dir results \
    --exp-name PuLID_4-Step_lightning_oneflow \
    --output-images \
    --ssim-threshold 0.8 \
    --baseline-dir results/PuLID_4-Step_lightning_baseline

python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_DIR/PuLID_ComfyUI/PuLID_IPAdapter_style_transfer.json \
    --exp-name PuLID_IPAdapter_style_transfer_oneflow \
    --ssim-threshold 0.8 \
    --output-images \
    --baseline-dir results/PuLID_IPAdapter_style_transfer_baseline

# Run InstantID workflow
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_DIR/ComfyUI_InstantID/instantid_posed_speedup.json \
    --exp-name instantid_posed_speedup \
    --output-images
