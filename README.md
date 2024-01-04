[![PyPI version](https://badge.fury.io/py/onediff.svg)](https://badge.fury.io/py/onediff)
[![Docker image build](https://github.com/siliconflow/onediff/actions/workflows/sd.yml/badge.svg)](https://github.com/siliconflow/onediff/actions/workflows/sd.yml)
[![Run examples](https://github.com/siliconflow/onediff/actions/workflows/examples.yml/badge.svg?event=schedule)](https://github.com/siliconflow/onediff/actions/workflows/examples.yml?query=event%3Aschedule)

<p align="center">
<img src="imgs/onediff_logo.png" height="100">
</p>

---

OneDiff is **an out-of-the-box acceleration library for diffusion models**  (especially for ComfyUI, HF diffusers, and Stable Diffusion web UI).

## Need help or talk
- GitHub Issues
- Discord of OneDiff: [![](https://dcbadge.vercel.app/api/server/RKJTjZMcPQ)](https://discord.gg/RKJTjZMcPQ)

## Easy to use
- Acceleration for popular UIs/libs
  - [ComfyUI](https://github.com/siliconflow/onediff/tree/main/onediff_comfy_nodes)
  - [HF diffusers 🤗](https://github.com/siliconflow/onediff/tree/main/examples)
  - [Stable Diffusion web UI](https://github.com/siliconflow/onediff/tree/main/onediff_sd_webui_extensions)
- Acceleration for state-of-the-art Models
  - [SDXL](https://github.com/siliconflow/onediff/blob/main/examples/text_to_image_sdxl.py)
  - [SDXL Turbo](https://github.com/siliconflow/onediff/blob/main/examples/text_to_image_sdxl_turbo.py)
  - [SD 1.5/2.1](https://github.com/siliconflow/onediff/blob/main/examples/text_to_image.py)
  - [LoRA (and dynamic switching LoRA)](https://github.com/siliconflow/onediff/blob/main/examples/text_to_image_sdxl_lora.py)
  - [ControlNet](https://github.com/siliconflow/onediff/blob/main/examples/text_to_image_controlnet.py)
  - [LCM](https://github.com/siliconflow/onediff/blob/main/examples/text_to_image_lcm.py) and [LCM LoRA](https://github.com/siliconflow/onediff/blob/main/examples/text_to_image_lcm_lora_sdxl.py)
  - [Stable Video Diffusion](https://github.com/siliconflow/onediff/blob/main/examples/image_to_video.py)
  - [DeepCache for ComfyUI](https://github.com/siliconflow/onediff/blob/8a35a9e7df45bbfa5bb05011b8357480acb5836e/onediff_comfy_nodes/_nodes.py#L414)
- Out-of-the-box acceleration
  - [ComfyUI Nodes](https://github.com/siliconflow/onediff/tree/main/onediff_comfy_nodes)
  - [Acceleration with oneflow_compile](https://github.com/siliconflow/onediff/blob/a38c5ea475c07b4527981ec5723ccac083ed0a9c/examples/text_to_image_sdxl.py#L53)
- Multi-resolution input
- Compile and save the compiled result offline, then load it online for serving
  - [Save and Load](https://github.com/siliconflow/onediff/blob/main/examples/text_to_image_sdxl_save_load.py)
  - [Change device to do multi-process serving](https://github.com/siliconflow/onediff/blob/main/examples/text_to_image_sdxl_mp_load.py)

## State-of-the-art performance
Updated on December 28, 2023.

<img src="benchmarks/pics/231228_sdxl_e2e_a100.png" width="600" height="400">

## OS and GPU support
- Linux
  - If you want to use OneDiff on Windows, please use it under WSL. 
- NVIDIA GPUs

## OneDiff Enterprise Edition
If you need **Enterprise Level Support** for your system or business, please send an email to business@siliconflow.com and tell us about your user case, deployment scale, and requirements.

OneDiff Enterprise Edition can be **subscripted for one month and one GPU** and the cost is low: https://siliconflow.com/product.html

|                      | OneDiff Enterprise   | OneDiff Community |
| -------------------- | ------------------- | ----------- |
| SD/SDXL series model Optimization| Yes | Yes|
| UNet/VAE/ControlNet Optimization | Yes      | Yes         |
| LoRA(and dynamic switching LoRA)                 | Yes             | Yes         |
| SDXL Turbo/LCM                  | Yes             | Yes         |
| Stable Video Diffusion |  Yes      | Yes |
| HF diffusers            | Yes                 | Yes         |
| ComfyUI              | Yes           | Yes         |
| Stable Diffusion web UI | Yes          | Yes         |
| Multiple Resolutions | Yes(No time cost for most of the cases)       | Yes(Cost a few seconds/minutes to compile for new input shape)           | 
| Technical Support for deployment    | High priority support       | Community           | 
| More Extreme and Dedicated optimization(usually another 20~50% performance gain)         |   Yes         |                 | 
| Support customized pipeline/workflow|           Yes              | |
| Get the latest technology/feature | Yes | |

## Install from source or Using in Docker
### Install from source

#### 1. Install OneFlow
> **_NOTE:_** We have updated OneFlow a lot for OneDiff, so please install OneFlow by the links below.

For CUDA 11.8
```
python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/community/cu118
```
<details>
<summary> Click to get OneFlow packages for other CUDA versions. </summary>
CUDA 12.1

```bash
python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/community/cu121
```

CUDA 12.2

```bash
python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/community/cu122
```

</details>


#### 2. Install torch and diffusers
```
python3 -m pip install "torch" "transformers==4.27.1" "diffusers[torch]==0.19.3"
```

#### 3. Install OneDiff
```
git clone https://github.com/siliconflow/onediff.git
cd onediff && python3 -m pip install -e .
```

#### 4. (Optional)Login huggingface-cli

```
python3 -m pip install huggingface_hub
 ~/.local/bin/huggingface-cli login
```

### Docker
```bash
docker pull oneflowinc/onediff:20231106
```
