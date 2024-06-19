
comfy_pid=3688922

python3 scripts/text_to_image.py -w  resources/oneflow/sdxl-control-lora-speedup.json --comfy-pid $comfy_pid # --output-images

# python3 scripts/text_to_image.py -w resources/baseline/ComfyUI_IPAdapter_plus/ipadapter_advanced.json --comfy-pid $comfy_pid  --output-images

python3 scripts/text_to_image.py -w resources/oneflow/ComfyUI_IPAdapter_plus/ipadapter_advanced.json --comfy-pid $comfy_pid  # --output-images

