# Run SD# with nexfort backend(Beta Release)


python3 onediff_diffusers_extensions/examples/sd3/text_to_image_sd3.py \
--compiler-config '{"mode": "quant:max-optimize:max-autotune:freezing:benchmark:low-precision:cudagraphs", \
    "memory_format": "channels_last"}' \
--quantize-config '{"quant_type": "fp8_e4m3_e4m3_dynamic_per_tensor"}'


TODO