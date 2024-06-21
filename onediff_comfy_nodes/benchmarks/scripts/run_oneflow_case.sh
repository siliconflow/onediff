
# python3 scripts/text_to_image.py -w  resources/oneflow/sdxl-control-lora-speedup.json  # --output-images

# python3 scripts/text_to_image.py -w resources/oneflow/ComfyUI_IPAdapter_plus/ipadapter_advanced.json # --output-images

# baseline 
python3 scripts/text_to_image.py \
    -w resources/baseline/lora.json resources/baseline/lora_multiple.json \
    --output-images 

python3 scripts/text_to_image.py \
    -w resources/oneflow/lora_speedup.json resources/oneflow/lora_multiple_speedup.json \
    --output-images