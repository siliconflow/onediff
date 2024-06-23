#!/bin/bash
set -e

STANDARD_OUTPUT=/share_nfs/hf_models/comfyui_resources/standard_output
COMFY_PORT=8188

# python3 scripts/text_to_image.py \
#     --comfy-port $COMFY_PORT \
#     -w resources/oneflow/sdxl-control-lora-speedup.json 

# # Baseline 
# python3 scripts/text_to_image.py \
#     -w resources/baseline/lora.json resources/baseline/lora_multiple.json \
#     --output-images 
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w resources/oneflow/lora_speedup.json resources/oneflow/lora_multiple_speedup.json \
    --baseline-dir $STANDARD_OUTPUT/test_lora_speedup 

# # Baseline 
# python3 scripts/text_to_image.py \
#      --comfy-port $COMFY_PORT \ 
#     -w resources/baseline/ComfyUI_IPAdapter_plus/ipadapter_advanced.json \
#     --output-images
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w resources/oneflow/ComfyUI_IPAdapter_plus/ipadapter_advanced.json \
    --output-images \
    --baseline-dir $STANDARD_OUTPUT/test_ipa



# # # # Baseline 
# # # python3 scripts/text_to_image.py \
# # #     --comfy-port $COMFY_PORT \
# # #     -w resources/baseline/ComfyUI_InstantID/instantid_posed.json \
# # #     --output-images
# python3 scripts/text_to_image.py \
#     --comfy-port $COMFY_PORT \
#     -w resources/oneflow/ComfyUI_InstantID/instantid_posed_speedup.json \
#     --output-images
# #     --baseline-dir /home/fengwen/worksplace/packages/onediff/onediff_comfy_nodes/benchmarks/results/exp/imgs
