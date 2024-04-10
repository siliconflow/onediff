
<p align="center">
<img src="../../../imgs/onediff_logo.png" height="100">
</p>

## <div align="center">OneDiff Quant 🚀 NEW Documentation</div>
OneDiff Enterprise offers a quantization method that reduces memory usage, increases speed, and maintains quality without any loss.

**Note**: Before proceeding with this document, please ensure you are familiar with the [OneDiff Community](./README.md) features and OneDiff ENTERPRISE  by referring to the  [ENTERPRISE Guide](https://github.com/siliconflow/onediff/blob/main/README_ENTERPRISE.md#install-onediff-enterprise)

1. [OneDiff Installation Guide](https://github.com/siliconflow/onediff/blob/main/README_ENTERPRISE.md#install-onediff-enterprise)
2. [OneDiffx Installation Guide](https://github.com/siliconflow/onediff/tree/main/onediff_diffusers_extensions#install-and-setup)

3.Install plotly
```bash
python3 -m pip install plotly 
```

## Online Quant

First, you need to download [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) model.
If you are a multi card user, please select the graphics card that executes the program
1. Set CUDA device using export CUDA_VISIBLE_DEVICES=7.

2. The log *.pt file is cached. Quantization result information can be found in `cache_dir`/quantization_stats.json.

# Baseline (non-optimized)
**Note: You can obtain the baseline by running the following command**

 ```bash
python onediff_diffusers_extensions/examples/text_to_image_online_quant.py \
        --model_id  /share_nfs/hf_models/stable-diffusion-xl-base-1.0  \
        --seed 1 \
        --backend torch  --height 1024 --width 1024 --output_file sdxl_torch.png
```


# OneDiff Quant(optimized)

**Note: You can run it using the following command**

 ```bash
python onediff_diffusers_extensions/examples/text_to_image_online_quant.py \
        --model_id  /share_nfs/hf_models/stable-diffusion-xl-base-1.0  \
        --seed 1 \
        --backend onediff \
        --cache_dir ./run_sdxl_quant \
        --height 1024 \
        --width 1024 \
        --output_file sdxl_quant.png   \
        --quantize \
        --conv_mae_threshold 0.1 \
        --linear_mae_threshold 0.2 \
        --conv_compute_density_threshold 900 \
        --linear_compute_density_threshold 300
```
| Option                                 | Range  | Default | Description                                                                  |
| -------------------------------------- | ------ | ------- | ---------------------------------------------------------------------------- |
| --conv_mae_threshold 0.1               | [0, 1] | 0.1     | MAE threshold for quantizing convolutional modules to 0.1.                   |
| --linear_mae_threshold 0.2             | [0, 1] | 0.2     | MAE threshold for quantizing linear modules to 0.2.                          |
| --conv_compute_density_threshold 900   | [0, ∞) | 900     | Computational density threshold for quantizing convolutional modules to 900. |
| --linear_compute_density_threshold 300 | [0, ∞) | 300     | Computational density threshold for quantizing linear modules to 300.        |

## Offline Quant

要load a quantized model. 实现自定义模型的量化，load a floating model that to be quantized as int8 请参考如下脚本


## Quant a custom model

To achieve quantization of custom models, please refer to the following script
```bash
python tests/test_quantize_custom_model.py
```

## Community and Support
[Here is the introduction of OneDiff Community.](https://github.com/siliconflow/onediff/wiki#onediff-community)
- [Create an issue](https://github.com/siliconflow/onediff/issues)
- Chat in Discord: [![](https://dcbadge.vercel.app/api/server/RKJTjZMcPQ?style=plastic)](https://discord.gg/RKJTjZMcPQ)
- Email for Enterprise Edition or other business inquiries: contact@siliconflow.com

- [How to use Online Quant](../../../onediff_diffusers_extensions/examples/text_to_image_online_quant.py)
- [How to use Offline Quant](./quantize_pipeline.py)
- [How to Quant a custom model](../../../tests/test_quantize_custom_model.py)
- [Community and Support](https://github.com/siliconflow/onediff?tab=readme-ov-file#community-and-support)