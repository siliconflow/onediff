### Default Configuration without LoRA

Run:
```
python3 onediff_diffusers_extensions/examples/quant_lora/test.py \
    --saved-image sd.png
```

Performance:
- Iterations per second: 7.03 it/s
- Time taken: 4.65 seconds
- Max used CUDA memory: 10.467 GiB


### Using LoRA

Run:
```
python3 onediff_diffusers_extensions/examples/quant_lora/test.py \
    --saved-image sd_lora.png \
    --use_lora
```

Performance:
- Iterations per second: 6.28 it/s
- Time taken: 5.16 seconds
- Max used CUDA memory: 10.481 GiB


### Compile

Run:
```
python3 onediff_diffusers_extensions/examples/quant_lora/test.py \
    --compiler-config '{"mode": "max-optimize:max-autotune:low-precision:cache-all", "memory_format": "channels_last"}' \
    --saved-image sd_lora_compile.png \
    --use_lora
```

Performance:
- Iterations per second: 13.29 it/s
- Time taken: 2.61 seconds
- Max used CUDA memory: 11.477 GiB


### Compiled with Quantization

Run:
```
python3 onediff_diffusers_extensions/examples/quant_lora/test.py \
    --compiler-config '{"mode": "quant:max-optimize:max-autotune:low-precision", "memory_format": "channels_last"}' \
    --quantize-config '{"quant_type": "int8_dynamic"}' \
    --saved-image sd_lora_int8.png \
    --use_lora
```

Performance:
- Iterations per second: 15.55 it/s
- Time taken: 2.22 seconds
- Max used CUDA memory: 8.804 GiB
