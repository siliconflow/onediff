#!/bin/bash

# 停止脚本在遇到错误时继续执行
set -e

cd /data/home/lijunliang/project/onediff
# 步骤1: 运行text_to_image.py脚本
echo "Running text_to_image.py..."
python onediff_diffusers_extensions/examples/text_to_image.py

# 步骤2: 进行模型量化
echo "Quantizing the model..."
python3 onediff_diffusers_extensions/tools/quantization/quantize-sd-fast.py \
    --model /share_nfs/hf_models/stable-diffusion-v1-5 \
    --quantized_model /data/home/lijunliang/project/model_pth \
    --height 512 \
    --width 512 \
    --conv_ssim_threshold 0.991 \
    --linear_ssim_threshold 0.991 \
    --conv_compute_density_threshold 0 \
    --linear_compute_density_threshold 0 \
    --save_as_float true \
    --cache_dir /data/home/lijunliang/project/cache

# 步骤3: 使用量化后的模型生成图片
echo "Generating image with quantized model..."
python onediff_diffusers_extensions/examples/text_to_image_sd_enterprise.py \
    --model /data/home/lijunliang/project/model_pth \
    --saved_image output_sd.png

# 步骤4: 运行SSIM比较脚本
echo "Comparing SSIM..."
python onediff_diffusers_extensions/tests/ssim_compare.py

echo "All processes completed successfully!"
