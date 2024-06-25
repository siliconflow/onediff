#!/bin/bash
set -e

COMFY_PORT=8188
STANDARD_OUTPUT=/share_nfs/hf_models/comfyui_resources/standard_output
WORKFLOW_DIR=resources/workflows/oneflow

# # # Baseline 
# # python3 scripts/text_to_image.py \
# #     --comfy-port $COMFY_PORT \
# #     -w resources/baseline/ComfyUI_InstantID/instantid_posed.json \
# #     --output-images
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_DIR/ComfyUI_InstantID/instantid_posed_speedup.json \
    --output-images