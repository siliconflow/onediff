# Online Quantization for ComfyUI

- [Install](#install)
- [Usage](#usage)
- [Performance Comparison](#performance-comparison)
  - [SDXL](#sdxl)
    - [Examples](#examples)
    - [Download the required model files](#download-the-required-model-files)
    - [Start ComfyUI](#start-comfyui)
  - [SD1.5](#sd15)
    - [Examples](#examples-1)
    - [Download the required model files](#download-the-required-model-files-1)
    - [Start ComfyUI](#start-comfyui-1)
  - [SVD](#svd)
    - [Examples](#examples-2)
    - [Download the required model files](#download-the-required-model-files-2)
    - [Start ComfyUI](#start-comfyui-2)
- [Parameter Description](#parameter-description)



## Install

1. [OneDiff Enterprise Installation Guide](https://github.com/siliconflow/onediff/blob/main/README_ENTERPRISE.md#install-onediff-enterprise)
2. [OneDiffx Installation Guide](https://github.com/siliconflow/onediff/tree/main/onediff_diffusers_extensions#install-and-setup)
3. Before use, please confirm with `python3 -m oneflow --doctor` to confirm the argument `enterprise: True`. If the information displayed is as follows `enterprise: True` then it meets the requirements. If the information displayed is as follows `enterprise: False`, then run the following command`pip uninstall oneflow onediff_quant -y`, and then follow the installation instructions for the Enterprise version to reinstall the OneDiff Enterprise version. You can find the relevant installation instructions through the following link: [OneDiff Enterprise Installation Instructions](https://github.com/siliconflow/onediff/blob/main/README_ENTERPRISE.md#install-onediff-enterprise)

## Usage:


Notes:

1. Specify the directory for saving graphs using `export COMFYUI_ONEDIFF_SAVE_GRAPH_DIR="/path/to/save/graphs"`.
2. When carrying out quantization for the first time, it is essential to analyze the data dependencies and identify the necessary parameters for quantification, such as the data's maximum and minimum values, which require additional computation time. Once these parameters are established and stored in cache, future quantization processes can directly utilize these parameters, thereby accelerating the processing speed. When quantization is performed a second time, the log file `*.pt` is cached. Information about the quantization results can be found in `cache_dir/quantization_stats.json`.

## Performance Comparison

### [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

We compared the performance of the stable-diffusion-xl-base-1.0 model in the three conditions and listed them in the subscript.

| Accelerator             | Baseline (non-optimized) | OneDiff(optimized) | OneDiff Quant(optimized) |
| ----------------------- | ------------------------ | ------------------ | ------------------------ |
| NVIDIA GeForce RTX 3090 | 5.63 s                   | 3.38 s ( ~40.0%)   | 2.60 s ( ~53.8%)         |

The following table shows the workflows used separately：

Note that you can download all images in this page and then drag or load them on ComfyUI to get the workflow embedded in the image.

#### Examples

| Baseline (non-optimized) | OneDiff(optimized) | OneDiff Quant(optimized) |
| ------------------------ | ------------------ | ------------------------ |
|![image](https://github.com/fmk345/pythonProject/assets/74238139/d5499822-e0b0-4186-831c-18f4c8921ec4)|![image](https://github.com/fmk345/pythonProject/assets/74238139/14feaaf4-6672-430c-85ff-d8c7d8b4d5a2)|![image](https://github.com/fmk345/pythonProject/assets/74238139/cb99f5d7-fb4f-421c-9783-a3adc4375759)|

Model parameters can be referred to [Parameter Description](#parameter-description).

#### Download the required model files


```
cd ComfyUI
wget -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
```



#### Start ComfyUI
```
python main.py --gpu-only
```





### SD1.5

We compared the performance of the stable-diffusion-v1-5 model in the three conditions and listed them in the subscript.

| Accelerator             | Baseline (non-optimized) | OneDiff(optimized) | OneDiff Quant(optimized) |
| ----------------------- | ------------------------ | ------------------ | ------------------------ |
| NVIDIA GeForce RTX 3090 | 3.54 s                   | 2.13 s ( ~39.8%)   | 1.85 s ( ~47.7%)         |

The following table shows the workflows used separately：

Note that you can download all images in this page and then drag or load them on ComfyUI to get the workflow embedded in the image.

#### Examples

| Baseline (non-optimized) | OneDiff(optimized) | OneDiff Quant(optimized) |
| ------------------------ | ------------------ | ------------------------ |
|![image](https://github.com/fmk345/pythonProject/assets/74238139/948271d2-24db-483f-9f33-81a64ae44c9e)|![image](https://github.com/fmk345/pythonProject/assets/74238139/08495a75-03f5-4e7d-93a2-b206ea755901)|![image](https://github.com/fmk345/pythonProject/assets/74238139/10c00186-fe72-4cca-92d2-2a672b8ffac5)|

Model parameters can be referred to [Parameter Description](#parameter-description).

#### Download the required model files



```
cd ComfyUI
wget -O  models/v1-5-pruned-emaonly.ckpt  https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt
```



#### Start ComfyUI
```
python main.py --gpu-only
```



### SVD

We compared the performance of the stable-diffusion-xl-base-1.0 model in the three conditions and listed them in the subscript.

| Accelerator             | Baseline (non-optimized) | OneDiff(optimized) | OneDiff Quant(optimized) |
| ----------------------- | ------------------------ | ------------------ | ------------------------ |
| NVIDIA A800-SXM4-80GB   | 35.54 s                  | 25.59 s (27.99 %)  | 22.30 s (37.25 %)        |


The following table shows the workflows used separately：

Note that you can download all images in this page and then drag or load them on ComfyUI to get the workflow embedded in the image.


#### Examples

| Baseline (non-optimized) | OneDiff(optimized) | OneDiff Quant(optimized) |
| ------------------------ | ------------------ | ------------------------ |
|![svd_baseline](https://github.com/siliconflow/onediff/assets/109639975/9c8871cf-088a-4606-8eae-e26994a08252)|![OneDiff(optimized)](https://github.com/siliconflow/onediff/assets/109639975/c8677c18-0d42-4ec0-8b1b-cb84e4c5aed9)|![OneDiff Quant(optimized)](https://github.com/siliconflow/onediff/assets/109639975/da8ff2d8-579e-42a5-b0db-390d20100889)|

Model parameters can be referred to [Parameter Description](#parameter-description).

#### Download the required model files



```
cd ComfyUI
wget -O  models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
wget -O  models/checkpoints/svd_xt_1_1.safetensors https://huggingface.co/vdo/stable-video-diffusion-img2vid-xt-1-1/resolve/main/svd_xt_1_1.safetensors
```



#### Start ComfyUI
```
python main.py --gpu-only
```

## Parameter Description
| Option                                 | Range  | Default | Description                                                                  |
| -------------------------------------- | ------ | ------- | ---------------------------------------------------------------------------- |
| quantized_conv_percentage                | [0, 100] | 100     |  Example value representing 100% quantization for linear layers     |
| quantized_linear_percentage           | [0, 100] | 100     | Example value representing 100% quantization for convolutional layers  |
| conv_compute_density_threshold    | [0, ∞) | 100     | Computational density threshold for quantizing convolutional modules to 100  |
| linear_compute_density_threshold  | [0, ∞) | 300     | Computational density threshold for quantizing linear modules to 300         |
