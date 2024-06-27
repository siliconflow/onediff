#!/bin/bash
set -e

COMFY_PORT=8188
STANDARD_OUTPUT=/share_nfs/hf_models/comfyui_resources/standard_output
WORKFLOW_DIR=resources/workflows/oneflow
WORKFLOW_BASE=resources/workflows/baseline
# # # # Baseline
# # # python3 scripts/text_to_image.py \
# # #     --comfy-port $COMFY_PORT \
# # #     -w resources/baseline/ComfyUI_InstantID/instantid_posed.json \
# # #     --output-images
# python3 scripts/text_to_image.py \
#     --comfy-port $COMFY_PORT \
#     -w $WORKFLOW_DIR/ComfyUI_InstantID/instantid_posed_speedup.json \
#     --output-images


########### PuLID_ComfyUI
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_BASE/PuLID_ComfyUI/PuLID_IPAdapter_style_transfer.json \
    --output-images 
    
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_DIR/PuLID_ComfyUI/PuLID_IPAdapter_style_transfer.json \
    --output-images \
    --baseline-dir /home/fengwen/worksplace/packages/onediff/onediff_comfy_nodes/benchmarks/results/exp/imgs
