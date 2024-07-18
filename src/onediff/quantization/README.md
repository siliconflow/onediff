
<p align="center">
<img src="../../../imgs/onediff_logo.png" height="100">
</p>

# <div align="center">OneDiff Quant 🚀 Documentation</div>
OneDiff Enterprise offers a quantization method that reduces memory usage, increases speed, and maintains quality without any loss.

Here's the optimized results, Timings for 30 steps in Diffusers-SDXL at 1024x1024.
| Accelerator             | Baseline (non-optimized) | OneDiff(optimized online) | OneDiff Quant(optimized) |
| ----------------------- | ------------------------ | ------------------ | ------------------------ |
| NVIDIA GeForce RTX 3090 | 8.03 s                   | 4.44 s ( ~44.7%)   | **3.34 s ( ~58.4%)**         |

- torch   {version: 2.2.1+cu121}
- oneflow {version: 0.9.1.dev20240406+cu121, enterprise: True}

Here's the optimized results, Timings for 30 steps in Diffusers-SD-1.5 at 1024x1024.
| Accelerator             | Baseline (non-optimized) | OneDiff(optimized online) | OneDiff Quant(optimized) |
| ----------------------- | ------------------------ | ------------------ | ------------------------ |
| NVIDIA GeForce RTX 3090 | 6.87 s                   | 3.41 s ( ~50.3%)   | **3.13 s ( ~54.4%)**         |

- torch   {version: 2.2.2+cu121}
- oneflow {version:  0.9.1.dev20240403+cu122, enterprise: True}

**Note**: Before proceeding with this document, please ensure you are familiar with the [OneDiff Community](../../../README.md) features and OneDiff ENTERPRISE  by referring to the  [ENTERPRISE Guide](../../../README_ENTERPRISE.md#install-onediff-enterprise).

- [Prepare environment](#prepare-environment)
- [Baseline (non-optimized)](#baseline-non-optimized)
- [How to use onediff quantization](#how-to-use-onediff-quantization)
  - [Online quantification](#online-quantification)
    - [Online quantification (optimized)](#online-quantification-optimized)
  - [Offline quantification](#offline-quantification)
- [Quantify a custom model](#quantify-a-custom-model)
- [Community and Support](#community-and-support)

## Prepare environment

You need to complete the following environment dependency installation.

- 1. [OneDiff Installation Guide](https://github.com/siliconflow/onediff/blob/main/README_ENTERPRISE.md#install-onediff-enterprise)
- 2. [OneDiffx Installation Guide](https://github.com/siliconflow/onediff/tree/main/onediff_diffusers_extensions#install-and-setup)

## Baseline (non-optimized)

You can obtain the baseline by running the following command.

```bash
python onediff_diffusers_extensions/examples/text_to_image_online_quant.py \
        --model_id  /PATH/TO/YOU/MODEL  \
        --seed 1 \
        --backend torch  --height 1024 --width 1024 --output_file sdxl_torch.png
```

## How to use onediff quantization

Onediff quantization supports acceleration of all diffusion models. This document will be explained based on the SDXL model.First, you can download [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) model.

### Online quantification

**Note**: When performing quantification for the first time, it is necessary to perform dependency analysis of the data and determine the parameters required for quantification, such as the maximum and minimum values of the data, which require additional computing time. Once these parameters are determined and cached, subsequent quantization processes can use these parameters directly, thus speeding up processing.When quantization is performed for the second time, the log `*.pt` file is cached. Quantization result information can be found in `cache_dir/quantization_stats.json`.

#### Online quantification (optimized)

You can run it using the following command.

```bash
python onediff_diffusers_extensions/examples/text_to_image_online_quant.py \
        --model_id  /PATH/TO/YOU/MODEL  \
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

The parameters of the preceding command are shown in the following table.
| Option                                 | Range  | Default | Description                                                                  |
| -------------------------------------- | ------ | ------- | ---------------------------------------------------------------------------- |
| --conv_mae_threshold 0.1               | [0, 1] | 0.1     | MAE threshold for quantizing convolutional modules to 0.1.                   |
| --linear_mae_threshold 0.2             | [0, 1] | 0.2     | MAE threshold for quantizing linear modules to 0.2.                          |
| --conv_compute_density_threshold 900   | [0, ∞) | 900     | Computational density threshold for quantizing convolutional modules to 900. |
| --linear_compute_density_threshold 300 | [0, ∞) | 300     | Computational density threshold for quantizing linear modules to 300.        |

### Offline quantification

To quantify a custom model as int8, run the following script.

```bash
python ./src/onediff/quantization/quant_pipeline_test.py \
        --floatting_model_path "stabilityai/stable-diffusion-xl-base-1.0" \
        --prompt "a photo of an astronaut riding a horse on mars" \
        --height 1024 \
        --width 1024 \
        --num_inference_steps 30 \
        --conv_compute_density_threshold 900 \
        --linear_compute_density_threshold 300 \
        --conv_ssim_threshold 0.985 \
        --linear_ssim_threshold 0.991 \
        --save_as_float False \
        --cache_dir "./run_sd-v1-5" \
        --quantized_model ./quantized_model
```

If you want to load a quantized model, you can modify the quantized_model parameter to the path of the specific model, such as the [sd-1.5-onediff-enterprise](https://huggingface.co/siliconflow/stable-diffusion-v1-5-onediff-comfy-enterprise-v1) and [sd-1.5-onediff-deepcache models](https://huggingface.co/siliconflow/stable-diffusion-v1-5-onediff-deepcache-int8). [Stable-diffusion-v2-1-onediff-enterprise](https://huggingface.co/siliconflow/stable-diffusion-v2-1-onediff-enterprise) it has not been quantified, so it needs to be quantified first.

```bash
python ./src/onediff/quantization/load_quantized_model.py \
        --prompt "a photo of an astronaut riding a horse on mars" \
        --height 1024 \
        --width 1024 \
        --num_inference_steps 30 \
        --quantized_model ./quantized_model
```

## Quantify a custom model

To achieve quantization of custom models, please refer to the following script.

```bash
python tests/test_quantize_custom_model.py
```

## Community and Support

[Here is the introduction of OneDiff Community.](https://github.com/siliconflow/onediff/wiki#onediff-community)
- [Create an issue](https://github.com/siliconflow/onediff/issues)
- Chat in Discord: [![](https://dcbadge.vercel.app/api/server/RKJTjZMcPQ?style=plastic)](https://discord.gg/RKJTjZMcPQ)
- Email for Enterprise Edition or other business inquiries: contact@siliconflow.com
