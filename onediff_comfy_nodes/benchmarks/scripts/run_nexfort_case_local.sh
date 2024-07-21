#!/bin/bash
set -e

COMFY_PORT=8188
export WORKFLOW_DIR=resources/workflows/nexfort
export WORKFLOW_BASIC=resources/workflows/baseline

# Run the SD3 baseline workflow
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_BASIC/sd3_basic.json \
    --output-dir results \
    --exp-name sd3_basic_baseline \
    --output-images

# Run the SD3 nexfort workflow
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_DIR/sd3_basic.json \
    --output-dir results \
    --exp-name sd3_basic_nexfort_warmup \
    --output-images \
    --baseline-dir results/sd3_basic_baseline


# Run the SD3 nexfort workflow
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_DIR/sd3_basic.json \
    --output-dir results \
    --exp-name sd3_basic_nexfort_infer \
    --output-images \
    --baseline-dir results/sd3_basic_baseline
