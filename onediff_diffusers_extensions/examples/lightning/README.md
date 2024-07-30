# Run SDXL-Lightning with OneDiff

## Environment Setup

### Set Up OneDiff
Follow the instructions to set up OneDiff from the https://github.com/siliconflow/onediff?tab=readme-ov-file#installation.

### Set Up Compiler Backend
OneDiff supports two compiler backends: OneFlow and NexFort. Follow the setup instructions for these backends from the https://github.com/siliconflow/onediff?tab=readme-ov-file#install-a-compiler-backend.


### Set Up SDXL-Lightning
- HF model: [SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning)
- HF pipeline: [diffusers usage](https://huggingface.co/ByteDance/SDXL-Lightning#2-step-4-step-8-step-unet)

## Compile

> [!NOTE]
Current test is based on an 8 steps distillation model.

### Run 1024x1024 Without Compile (Original PyTorch HF Diffusers Baseline)
```bash
python3 onediff_diffusers_extensions/examples/lightning/text_to_image_sdxl_light.py \
--saved_image sdxl_light.png
```

### Run 1024x1024 With Compile [OneFlow Backend]
```bash
python3 onediff_diffusers_extensions/examples/lightning/text_to_image_sdxl_light.py \
--compiler oneflow \
--saved_image sdxl_light_oneflow_compile.png
```

### Run 1024x1024 With Compile [NexFort Backend]
```bash
python3 onediff_diffusers_extensions/examples/lightning/text_to_image_sdxl_light.py \
--compiler nexfort \
--compiler-config '{"mode": "max-optimize:max-autotune:low-precision", "memory_format": "channels_last", "options": {"triton.fuse_attention_allow_fp16_reduction": false}}' \
--saved_image sdxl_light_nexfort_compile.png
```


## Quantization (Int8)

> [!NOTE]
Quantization is a feature for onediff enterprise.

### Run 1024x1024 With Quantization [OneFlow Backend]

Execute the following command to quantize the model, where `--quantized_model` is the path to the quantized model. For an introduction to the quantization parameters, refer to: https://github.com/siliconflow/onediff/blob/main/README_ENTERPRISE.md#diffusers-with-onediff-enterprise

```
python3 onediff_diffusers_extensions/tools/quantization/quantize-sd-fast.py \
   --quantized_model ./sdxl_lightning_oneflow_quant \
   --conv_ssim_threshold 0.1 \
   --linear_ssim_threshold 0.1 \
   --conv_compute_density_threshold 300 \
   --linear_compute_density_threshold 300 \
   --save_as_float true \
   --use_lightning 1
```

Test the quantized model:

```
python3 onediff_diffusers_extensions/examples/lightning/text_to_image_sdxl_light.py \
--compiler oneflow \
--use_quantization \
--base ./sdxl_lightning_oneflow_quant \
--saved_image sdxl_light_oneflow_quant.png
```


### Run 1024x1024 With Quantization [NexFort Backend]

```
python3 onediff_diffusers_extensions/examples/lightning/text_to_image_sdxl_light.py \
  --compiler nexfort \
  --compiler-config '{"mode": "max-optimize:max-autotune:low-precision", "memory_format": "channels_last", "options": {"triton.fuse_attention_allow_fp16_reduction": false}}' \
  --use_quantization \
  --quantize-config '{"quant_type": "int8_dynamic"}' \
  --saved_image sdxl_light_nexfort_quant.png
```


## Performance Comparison

**Testing on an NVIDIA RTX 4090 GPU, using a resolution of 1024x1024 and 8 steps:**

| Configuration             | Iteration Speed (it/s)          | E2E Time (seconds)              |
|---------------------------|---------------------------------|---------------------------------|
| PyTorch                   | 14.68                           | 0.840                           |
| OneFlow Compile           | 29.06 (+97.83%)                 | 0.530 (-36.90%)                 |
| OneFlow Quantization      | 43.45 (+195.95%)                | 0.424 (-49.52%)                 |
| NexFort Compile           | 28.07 (+91.18%)                 | 0.526 (-37.38%)                 |
| NexFort Quantization      | 30.85 (+110.15%)                | 0.476 (-43.33%)                 |
