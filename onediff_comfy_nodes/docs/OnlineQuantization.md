## Online quantization for comfyui

### Install

1. [OneDiff Installation Guide](https://github.com/siliconflow/onediff/blob/main/README_ENTERPRISE.md#install-onediff-enterprise)
2. [OneDiffx Installation Guide](https://github.com/siliconflow/onediff/tree/main/onediff_diffusers_extensions#install-and-setup)

## Usage:
| Option                                 | Range  | Default | Description                                                                  |
| -------------------------------------- | ------ | ------- | ---------------------------------------------------------------------------- |
| quantized_conv_percentage                | [0, 100] | 100     |  Example value representing 70% quantization for linear layers｜
| quantized_linear_percentage           | [0, 100] | 100     | Example value representing 100% quantization for convolutional layers  |
| conv_compute_density_threshold    | [0, ∞) | 100     | Computational density threshold for quantizing convolutional modules to 100. |
| linear_compute_density_threshold  | [0, ∞) | 300     | Computational density threshold for quantizing linear modules to 300.        |

Notes:

1. Specify the directory for saving graphs using export COMFYUI_ONEDIFF_SAVE_GRAPH_DIR="/path/to/save/graphs".
2. The log *.pt file is cached. Quantization result information can be found in `cache_dir`/quantization_stats.json.

## Performance Comparison

### [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
Updated on Mon 08 Apr 2024

![quant_sdxl](https://github.com/siliconflow/onediff/assets/109639975/b8f8da75-944b-4553-aea3-69c19886af37)

| Accelerator             | Baseline (non-optimized) | OneDiff(optimized) | OneDiff Quant(optimized) |
| ----------------------- | ------------------------ | ------------------ | ------------------------ |
| NVIDIA GeForce RTX 3090 | 8.03 s                   | 4.44 s ( ~44.7%)   | 3.34 s ( ~58.4%)         |

- torch   `python -c "import torch; print(torch.__version__)"`: {version: 2.2.1+cu121}
- oneflow  `python -m oneflow --doctor`: {git_commit: 710818c, version: 0.9.1.dev20240406+cu121, enterprise: True}
- Start comfyui command: `python main.py --gpu-only`

### SD1.5

![sd 1.5 ](https://github.com/siliconflow/onediff/assets/109639975/49a8ab1b-e2be-4719-a962-33b813f5e83f)

### SVD
![svd_quant](https://github.com/siliconflow/onediff/assets/109639975/93ebe3d5-8413-4a7e-8b93-fd016f61abe9)

| Accelerator             | Baseline (non-optimized) | OneDiff(optimized) | OneDiff Quant(optimized) |
| ----------------------- | ------------------------ | ------------------ | ------------------------ |
| NVIDIA A800-SXM4-80GB   | 63.64 s                   | 47.57 s ( ~25.25%)   | 49.83 s ( ~21.7%)     |

- torch   `python -c "import torch; print(torch.__version__)"`: {version: 2.4.0.dev20240405+cu121}
- oneflow  `python -m oneflow --doctor`: {git_commit: 4ed3138, version: 0.9.1.dev20240402+cu122, enterprise: True}
- ComfyUI Tue Apr 9 commit: 4201181b35402e0a992b861f8d2f0e0b267f52fa
- Start comfyui command: `python main.py --gpu-only`
- Python 3.10.13
