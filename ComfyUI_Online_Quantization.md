## Online quantization for comfyui

### Install

1. [OneDiff Installation Guide](https://github.com/siliconflow/onediff/blob/main/README_ENTERPRISE.md#install-onediff-enterprise)
2. [OneDiffx Installation Guide](https://github.com/siliconflow/onediff/tree/main/onediff_diffusers_extensions#install-and-setup)
3. [ComfyUI with OneDiff Enterprise](https://github.com/siliconflow/onediff/blob/main/README_ENTERPRISE.md#comfyui-with-onediff-enterprise)
3. Before use, please confirm with `$ python -m oneflow --doctor` to confirm the argument `enterprise: True`. If the argument `enterprise: False`, then run the following command`pip uninstall oneflow -y  && pip uninstall onediff_quant -y`, and then follow the installation instructions for the Enterprise version to reinstall the OneDiff Enterprise version. You can find the relevant installation instructions through the following link: [OneDiff Enterprise Installation Instructions](https://github.com/siliconflow/onediff/blob/main/README_ENTERPRISE.md#install-onediff-enterprise)

## Usage:
| Option                                 | Range  | Default | Description                                                                  |
| -------------------------------------- | ------ | ------- | ---------------------------------------------------------------------------- |
| quantized_conv_percentage                | [0, 100] | 100     |  Example value representing 100% quantization for linear layers｜
| quantized_linear_percentage           | [0, 100] | 100     | Example value representing 100% quantization for convolutional layers  |
| conv_compute_density_threshold    | [0, ∞) | 100     | Computational density threshold for quantizing convolutional modules to 100. |
| linear_compute_density_threshold  | [0, ∞) | 300     | Computational density threshold for quantizing linear modules to 300.        |

Notes:

1. Specify the directory for saving graphsusing export COMFYUI_ONEDIFF_SAVE_GRAPH_DIR="/path/to/save/graphs".
2. The log *.pt file is cached. Quantization result information can be found in `cache_dir`/quantization_stats.json.

## Performance Comparison

### [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

| Accelerator             | Baseline (non-optimized) | OneDiff(optimized) | OneDiff Quant(optimized) |
| ----------------------- | ------------------------ | ------------------ | ------------------------ |
| NVIDIA GeForce RTX 3090 | 5.63 s                   | 3.38 s ( ~40.0%)   | 2.60 s ( ~53.8%)         |

The following table shows the workflows used separately：

Note that you can download all images in this page and then drag or load them on ComfyUI to get the workflow embedded in the image.

### Examples

| Baseline (non-optimized) | OneDiff(optimized) | OneDiff Quant(optimized) |
| ------------------------ | ------------------ | ------------------------ |
|![image](https://github.com/fmk345/pythonProject/assets/74238139/d5499822-e0b0-4186-831c-18f4c8921ec4)|![image](https://github.com/fmk345/pythonProject/assets/74238139/14feaaf4-6672-430c-85ff-d8c7d8b4d5a2)|![image](https://github.com/fmk345/pythonProject/assets/74238139/cb99f5d7-fb4f-421c-9783-a3adc4375759)|

<details open>
<summary> Download the required model files </summary>


##### For NA/EU users
```
cd ComfyUI

wget -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
```

##### For CN users
```
cd ComfyUI

wget -O models/checkpoints/sd_xl_base_1.0.safetensors https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
```

</details>

#### Environment
- torch : {version: 2.2.1+cu121}
- oneflow  : {git_commit: 8e20ea9, version: 0.9.1.dev20240410+cu122, enterprise: True}
- ComfyUI Tue Apr 9 commit: 4201181b35402e0a992b861f8d2f0e0b267f52fa
- Start comfyui command: `python main.py --gpu-only`
- Python 3.10.13

### SD1.5

| Accelerator             | Baseline (non-optimized) | OneDiff(optimized) | OneDiff Quant(optimized) |
| ----------------------- | ------------------------ | ------------------ | ------------------------ |
| NVIDIA GeForce RTX 3090 | 3.54 s                   | 2.13 s ( ~39.8%)   | 1.85 s ( ~47.7%)         |

The following table shows the workflows used separately：

Note that you can download all images in this page and then drag or load them on ComfyUI to get the workflow embedded in the image.

| Baseline (non-optimized) | OneDiff(optimized) | OneDiff Quant(optimized) |
| ------------------------ | ------------------ | ------------------------ |
|![image](https://github.com/fmk345/pythonProject/assets/74238139/948271d2-24db-483f-9f33-81a64ae44c9e)|![image](https://github.com/fmk345/pythonProject/assets/74238139/08495a75-03f5-4e7d-93a2-b206ea755901)|![image](https://github.com/fmk345/pythonProject/assets/74238139/10c00186-fe72-4cca-92d2-2a672b8ffac5)|

<details open>
<summary> Download the required model files </summary>


##### For NA/EU users
```
cd ComfyUI

wget -O  models/v1-5-pruned-emaonly.ckpt  https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt

```

##### For CN users
```
cd ComfyUI

wget -O  models/v1-5-pruned-emaonly.ckpt  https://hf-mirror.com/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
```

</details>

#### Environment
- torch : {version: 2.2.1+cu121}
- oneflow  : {git_commit: 8e20ea9, version: 0.9.1.dev20240410+cu122, enterprise: True}
- ComfyUI Tue Apr 9 commit: 4201181b35402e0a992b861f8d2f0e0b267f52fa
- Start comfyui command: `python main.py --gpu-only`
- Python 3.10.13

### SVD

| Accelerator             | Baseline (non-optimized) | OneDiff(optimized) | OneDiff Quant(optimized) |
| ----------------------- | ------------------------ | ------------------ | ------------------------ |
| NVIDIA A800-SXM4-80GB   | 35.54 s                  | 25.59 s (27.99 %)  | 22.30 s (37.25 %)        |


The following table shows the workflows used separately：

Note that you can download all images in this page and then drag or load them on ComfyUI to get the workflow embedded in the image.

| Baseline (non-optimized) | OneDiff(optimized) | OneDiff Quant(optimized) |
| ------------------------ | ------------------ | ------------------------ |
|![svd_baseline](https://github.com/siliconflow/onediff/assets/109639975/9c8871cf-088a-4606-8eae-e26994a08252)|![OneDiff(optimized)](https://github.com/siliconflow/onediff/assets/109639975/c8677c18-0d42-4ec0-8b1b-cb84e4c5aed9)|![OneDiff Quant(optimized)](https://github.com/siliconflow/onediff/assets/109639975/da8ff2d8-579e-42a5-b0db-390d20100889)|


<details open>
<summary> Download the required model files </summary>


##### For NA/EU users
```
cd ComfyUI

wget -O  models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

wget -O  models/checkpoints/svd_xt_1_1.safetensors https://huggingface.co/vdo/stable-video-diffusion-img2vid-xt-1-1/resolve/main/svd_xt_1_1.safetensors
```

##### For CN users
```
cd ComfyUI

wget -O  models/checkpoints/sd_xl_base_1.0.safetensors https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

wget -O  models/checkpoints/svd_xt_1_1.safetensors https://hf-mirror.com/vdo/stable-video-diffusion-img2vid-xt-1-1/resolve/main/svd_xt_1_1.safetensors
```

</details>

#### Environment
- torch : {version: 2.2.1+cu121}
- oneflow  : {git_commit: 8e20ea9, version: 0.9.1.dev20240410+cu122, enterprise: True}
- ComfyUI Tue Apr 9 commit: 4201181b35402e0a992b861f8d2f0e0b267f52fa
- Start comfyui command: `python main.py --gpu-only`
- Python 3.10.13