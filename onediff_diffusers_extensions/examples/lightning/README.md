Run:    
    

```
python3 onediff_diffusers_extensions/tools/quantization/quantize-sd-fast.py \
   --quantized_model ./sdxl_lightning_quant \
   --conv_ssim_threshold 0.1 \
   --linear_ssim_threshold 0.1 \
   --conv_compute_density_threshold 900 \
   --linear_compute_density_threshold 300 \
   --save_as_float true \
   --use_lightning 1
```