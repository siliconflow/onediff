<p align="center">
<img src="imgs/onediff_logo.png" height="100">
</p>

---
<p align="center">
  <a href="https://pypistats.org/packages/onediff" target="_blank"><img src="https://img.shields.io/pypi/dw/onediff?style=square&label=Pip install"></a>
  <a href="https://github.com/siliconflow/onediff/stargazers" target="_blank"><img src="https://img.shields.io/github/stars/siliconflow/onediff?style=square&label=Stars&color=green"></a>
  <a href="https://github.com/siliconflow/onediff?tab=Apache-2.0-1-ov-file#readme" target="_blank"><img src="https://img.shields.io/github/license/siliconflow/onediff"></a>
  <a href="https://github.com/siliconflow/onediff/wiki" target="_blank"><img src="https://img.shields.io/badge/OneDiff-Community wiki-green"></a>
  <a href="https://twitter.com/search?q=%22onediff%22&src=typed_query&f=live" target="_blank"><img src="https://img.shields.io/badge/Twitter-Discuss-green?logo=twitter&amp"></a>
  <a href="https://discord.gg/RKJTjZMcPQ" target="_blank"><img src="https://dcbadge.vercel.app/api/server/RKJTjZMcPQ?style=square"></a>
</p>

<p align="center">
  <a href="https://github.com/siliconflow/onediff/milestone/3" target="_blank"><img src="https://img.shields.io/github/milestones/progress/siliconflow/onediff/3"></a>
  <a href="https://github.com/siliconflow/onediff/issues?q=is%3Aopen+is%3Aissue" target="_blank"><img src="https://img.shields.io/github/issues/siliconflow/onediff"></a>
  <a href="https://github.com/siliconflow/onediff/issues?q=is%3Aissue+is%3Aclosed" target="_blank"><img src="https://img.shields.io/github/issues-closed/siliconflow/onediff?color=blue"></a>
  <a href="https://github.com/siliconflow/onediff/actions/workflows/sd.yml" target="_blank"><img src="https://github.com/siliconflow/onediff/actions/workflows/sd.yml/badge.svg"></a>
  <a href="https://github.com/siliconflow/onediff/actions/workflows/examples.yml?query=event%3Aschedule" target="_blank"><img src="https://github.com/siliconflow/onediff/actions/workflows/examples.yml/badge.svg?event=schedule"></a>
</p>

## OneDiff
OneDiff is an out-of-the-box acceleration library for diffusion models, it provides:
- PyTorch Module compilation tools and strong optimized GPU Kernels for diffusion models
- Out-of-the-box acceleration for popular UIs/libs
  - [OneDiff for HF diffusers 🤗](https://github.com/siliconflow/onediff/tree/main/onediff_diffusers_extensions)
  - [OneDiff for ComfyUI](https://github.com/siliconflow/onediff/tree/main/onediff_comfy_nodes)
  - [OneDiff for Stable Diffusion web UI](https://github.com/siliconflow/onediff/tree/main/onediff_sd_webui_extensions)

For example:

<img src="imgs/replace_a100.png" height="300">

OneDiff is the abbreviation of "**one** line of code to accelerate **diff**usion models".

## News
The latest news:
- :rocket:[OneDiff 1.0 is out! (Acceleration of SD & SVD with one line of code)](https://www.reddit.com/r/StableDiffusion/comments/1c5gy1e/onediff_10_is_out_acceleration_of_sd_svd_with_one/)
- :rocket:[Accelerating Stable Video Diffusion 3x faster with OneDiff DeepCache + Int8](https://www.reddit.com/r/StableDiffusion/comments/1adu2hn/accelerating_stable_video_diffusion_3x_faster/)
- :rocket:[Accelerating SDXL 3x faster with DeepCache and OneDiff](https://www.reddit.com/r/StableDiffusion/comments/18lz2ir/accelerating_sdxl_3x_faster_with_deepcache_and/)

## Community and Support
[Here is the introduction of OneDiff Community.](https://github.com/siliconflow/onediff/wiki#onediff-community)
- [Create an issue](https://github.com/siliconflow/onediff/issues)
- Chat in Discord: [![](https://dcbadge.vercel.app/api/server/RKJTjZMcPQ?style=plastic)](https://discord.gg/RKJTjZMcPQ)
- Email for Enterprise Edition or other business inquiries: contact@siliconflow.com

## OS and GPU Compatibility
- Linux
  - If you want to use OneDiff on Windows, please use it under WSL.
  - [The guide to install OneDiff in WSL2](https://github.com/siliconflow/onediff/wiki/Run-OneDiff-on-Windows-by-WSL2).

- NVIDIA GPUs
  - [Compatibility with Nvidia GPUs](https://github.com/siliconflow/onediff/wiki/Compatibility-with-Nvidia-GPUs).

---
The Full Introduction of OneDiff:
<!-- toc -->
- [About OneDiff](#about-onediff)
  - [Architecture](#architecture)
  - [State-of-the-art performance](#state-of-the-art-performance)
  - [Features](#features)
  - [Acceleration for State-of-the-art models](#acceleration-for-state-of-the-art-models)
  - [Acceleration for production environment](#acceleration-for-production-environment)
  - [OneDiff Quality Evaluation](#onediff-quality-evaluation)
  - [OneDiff Enterprise Edition](#onediff-enterprise-edition)
- [Installation](#installation)
- [Release](#release)
<!-- tocstop -->

## About OneDiff
### Architecture

OneDiff interfaces with various front-end sd frameworks upward, and uses a custom virtual machine mixed with PyTorch as the inference engine downward.

<img src="imgs/onediff_arch.png" height="500">

### State-of-the-art performance
#### SDXL E2E time
- Model stabilityai/stable-diffusion-xl-base-1.0;
- Image size 1024*1024, batch size 1, steps 30;
- NVIDIA A100 80G SXM4;

<img src="imgs/0_12_sdxl.png" height="400">

#### SVD E2E time
- Model stabilityai/stable-video-diffusion-img2vid-xt;
- Image size 576*1024, batch size 1, steps 25, decoder chunk size 5;
- NVIDIA A100 80G SXM4;

<img src="imgs/0_12_svd.png" height="400">

Note that we haven't got the way to run SVD with TensorRT on Feb 29 2024.

### Features

| Functionality | Details |
|----------------|----------------------------|
| Compiling Time   | About 1 minute (SDXL) |
| Deployment Methods              | Plug and Play |
| Dynamic Image Size Support  | Support with no overhead |
| Model Support                 | SD1.5~2.1, SDXL, SDXL Turbo, etc. |
| Algorithm Support             | SD standard workflow, LoRA, ControlNet, SVD, InstantID, SDXL Lightning, etc. |
| SD Framework Support | ComfyUI, Diffusers, SD-webui |
| Save & Load Accelerated Models | Yes |
| Time of LoRA Switching | Hundreds of milliseconds |
| LoRA Occupancy | Tens of MB to hundreds of MB. |
| Device Support | NVIDIA GPU 3090 RTX/4090 RTX/A100/A800/A10 etc. (Compatibility with Ascend in progress) |


### Acceleration for State-of-the-art models
OneDiff supports the acceleration for SOTA models.
* stable: release for public usage, and has long-term support;
* beta: release for professional usage, and has long-term support;
* alpha: early release for expert usage, and should be careful to use;

| AIGC Type | Models                      | HF diffusers |            | ComfyUI   |            | SD web UI |            |
| --------- | --------------------------- | ------------ | ---------- | --------- | ---------- | --------- | ---------- |
|           |                             | Community    | Enterprise | Community | Enterprise | Community | Enterprise |
| Image     | SD 1.5                      | stable       | stable     | stable    | stable     | stable    | stable     |
|           | SD 2.1                      | stable       | stable     | stable    | stable     | stable    | stable     |
|           | SDXL                        | stable       | stable     | stable    | stable     | stable    | stable     |
|           | LoRA                        | stable       |            | stable    |            | stable    |            |
|           | ControlNet                  | stable       |            | stable    |            |           |            |
|           | SDXL Turbo                  | stable       |            | stable    |            |           |            |
|           | LCM                         | stable       |            | stable    |            |           |            |
|           | SDXL DeepCache              | alpha        | alpha      | alpha     | alpha      |           |            |
|           | InstantID                   | beta         |            | beta      |            |           |            |
| Video     | SVD(stable Video Diffusion) | stable       | stable     | stable    | stable     |           |            |
|           | SVD DeepCache               | alpha        | alpha      | alpha     | alpha      |           |            |

**Note: Enterprise Edition contains all the functionality in Community Edition.**

### Acceleration for production environment
#### PyTorch Module compilation
- [compilation with oneflow_compile](https://github.com/siliconflow/onediff/blob/main/onediff_diffusers_extensions/examples/text_to_image_sdxl.py)
#### Avoid compilation time for new input shape
- [Support Multi-resolution input](https://github.com/siliconflow/onediff/blob/main/onediff_diffusers_extensions/examples/text_to_image_sdxl.py)
#### Avoid compilation time for online serving
Compile and save the compiled result offline, then load it online for serving
- [Save and Load the compiled graph](https://github.com/siliconflow/onediff/blob/main/onediff_diffusers_extensions/examples/text_to_image_sdxl_save_load.py)
- [Change device of the compiled graph to do multi-process serving](https://github.com/siliconflow/onediff/blob/main/onediff_diffusers_extensions/examples/text_to_image_sdxl_mp_load.py)
  - Compile at one device(such as device 0), then use the compiled result to other device(such as device 1~7).

### OneDiff Quality Evaluation

We also maintain a repository for benchmarking the quality of generation after acceleration using OneDiff:
[OneDiffGenMetrics](https://github.com/siliconflow/OneDiffGenMetrics)

### OneDiff Enterprise Edition
If you need **Enterprise-level Support** for your system or business, you can
- Subscribe to the OneDiff Enterprise Edition directly through our website. Upon purchase, you'll gain immediate access to comprehensive support: https://siliconflow.com/onediff.html
- For a more personalized approach, please email us at contact@siliconflow.com. Include details about your use case, deployment size, and any specific needs you might have.

The OneDiff Enterprise Edition is available for a monthly subscription and is designed to be cost-effective, even for systems utilizing a **single GPU**.

|                                                                                                           | OneDiff Enterprise Edition              | OneDiff Community Edition               |
| --------------------------------------------------------------------------------------------------------- | --------------------------------------- | --------------------------------------- |
| Multiple Resolutions                                                                                      | Yes(No time cost for most of the cases) | Yes(No time cost for most of the cases) |
| More Extreme and Dedicated optimization(usually another 20~100% performance gain) for the most used model | Yes                                     |                                         |
| Tools for specific(very large scale) server side deployment                                               | Yes                                     |                                         |
| Technical Support for deployment                                                                          | High priority support                   | Community                               |
| Get the experimental features                                                                             | Yes                                     |                                         |

## Installation
### OneDiff Installation

#### 1. Install OneFlow
> **_NOTE:_** We have updated OneFlow frequently for OneDiff, so please install OneFlow by the links below.

- **CUDA 11.8**

  For NA/EU users
  ```bash
  python3 -m pip install -U --pre oneflow -f https://github.com/siliconflow/oneflow_releases/releases/expanded_assets/community_cu118
  ```

  For CN users
  ```bash
  python3 -m pip install -U --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/community/cu118
  ```


<details>
<summary> Click to get OneFlow packages for other CUDA versions. </summary>

- **CUDA 12.1**

  For NA/EU users
  ```bash
  python3 -m pip install -U --pre oneflow -f https://github.com/siliconflow/oneflow_releases/releases/expanded_assets/community_cu121
  ```

  For CN users
  ```bash
  python3 -m pip install -U --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/community/cu121
  ```


- **CUDA 12.2**

  For NA/EU users
  ```bash
  python3 -m pip install -U --pre oneflow -f https://github.com/siliconflow/oneflow_releases/releases/expanded_assets/community_cu122
  ```
  For CN users
  ```bash
  python3 -m pip install -U --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/community/cu122
  ```



</details>


#### 2. Install torch and diffusers
**Note: You can choose the latest versions you want for diffusers or transformers.**
```
python3 -m pip install "torch" "transformers==4.27.1" "diffusers[torch]==0.19.3"
```


#### 3. Install OneDiff

- From PyPI
```
python3 -m pip install --pre onediff
```
- From source
```
git clone https://github.com/siliconflow/onediff.git
cd onediff && python3 -m pip install -e .
```

> **_NOTE:_** If you intend to utilize plugins for ComfyUI/StableDiffusion-WebUI, we highly recommend installing OneDiff from the source rather than PyPI. This is necessary as you'll need to manually copy (or create a soft link) for the relevant code into the extension folder of these UIs/Libs.

#### 4. (Optional)Login huggingface-cli

```bash
python3 -m pip install huggingface_hub
huggingface-cli login
```
