python3 onediff_diffusers_extensions/examples/quant_lora/test.py \
    --saved-image sd.png

python3 onediff_diffusers_extensions/examples/quant_lora/test.py \
    --saved-image sd_lora.png \
    --use_lora

python3 onediff_diffusers_extensions/examples/quant_lora/test.py \
    --compiler-config '{"mode": "max-optimize:max-autotune:low-precision:cache-all", "memory_format": "channels_last"}' \
    --saved-image sd_lora_compile.png \
    --use_lora

python3 onediff_diffusers_extensions/examples/quant_lora/test.py \
    --compiler-config '{"mode": "quant:max-optimize:max-autotune:low-precision", "memory_format": "channels_last"}' \
    --quantize-config '{"quant_type": "fp8_e4m3_e4m3_dynamic_per_tensor"}' \
    --saved-image sd_lora_fp8.png \
    --use_lora
