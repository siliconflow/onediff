<p align="center">
<img src="imgs/onediff_logo.png" height="100">
</p>

---

[![Docker image build](https://github.com/siliconflow/onediff/actions/workflows/sd.yml/badge.svg)](https://github.com/siliconflow/onediff/actions/workflows/sd.yml)
[![Run examples](https://github.com/siliconflow/onediff/actions/workflows/examples.yml/badge.svg?event=schedule)](https://github.com/siliconflow/onediff/actions/workflows/examples.yml?query=event%3Aschedule)


OneDiff is an out-of-the-box acceleration library for diffusion models, it provides:
- PyTorch Module compilation tools and strong optimized GPU Kernels for diffusion models
- Out-of-the-box acceleration for popular UIs/libs
  - [OneDiff for HF diffusers 🤗](https://github.com/siliconflow/onediff/tree/main/onediff_diffusers_extensions)
  - [OneDiff for ComfyUI](https://github.com/siliconflow/onediff/tree/main/onediff_comfy_nodes)
  - [OneDiff for Stable Diffusion web UI](https://github.com/siliconflow/onediff/tree/main/onediff_sd_webui_extensions)

OneDiff is the abbreviation of "**one** line of code to accelerate **diff**usion models". Here is the latested news:

- :rocket:[Accelerating Stable Video Diffusion 3x faster with OneDiff DeepCache + Int8](https://www.reddit.com/r/StableDiffusion/comments/1adu2hn/accelerating_stable_video_diffusion_3x_faster/)
- :rocket:[Accelerating SDXL 3x faster with DeepCache and OneDiff](https://www.reddit.com/r/StableDiffusion/comments/18lz2ir/accelerating_sdxl_3x_faster_with_deepcache_and/)
- :rocket:[InstantID can run 1.8x Faster with OneDiff](https://www.reddit.com/r/StableDiffusion/comments/1al19ek/instantid_can_run_18x_faster_with_onediff/)

The Full introduction of OneDiff:
<!-- toc -->
- [More About OneDiff](#more-about-onediff)
  - [State-of-the-art performance](#state-of-the-art-performance)
  - [Acceleration for production environment](#acceleration-for-production-environment)
  - [Acceleration for State-of-the-art models](#acceleration-for-state-of-the-art-models)
  - [OneDiff Enterprise Edition](#onediff-enterprise-edition)
  - [Roadmap](#roadmap)
- [Community and Support](#community-and-support)
- [Installation](#installation)
- [Release](#release)

<!-- tocstop -->

## More About OneDiff

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

### Acceleration for State-of-the-art models
OneDiff support the acceleratioin for SOTA models.
| AIGC Type | Models                      | HF diffusers |            | ComfyUI   |            | SD web UI |            |
| --------- | --------------------------- | ------------ | ---------- | --------- | ---------- | --------- | ---------- |
|           |                             | Community    | Enterprise | Community | Enterprise | Community | Enterprise |
| Image     | SD 1.5                      | stable       | stable     | stable    | stable     | beta      | beta       |
|           | SD 2.1                      | stable       | stable     | stable    | stable     | beta      | beta       |
|           | SDXL                        | stable       | stable     | stable    | stable     | beta      | beta       |
|           | LoRA                        | stable       |            | stable    |            | beta      |            |
|           | ControlNet                  | stable       |            | stable    |            |           |            |
|           | SDXL Turbo                  | stable       |            | stable    |            |           |            |
|           | LCM                         | stable       |            | stable    |            |           |            |
|           | SDXL DeepCache              | stable       | beta       | stable    | beta       |           |            |
|           | InstantID                   | stable       |            | stable    |            |           |            |
| Video     | SVD(stable Video Diffusion) | stable       | beta       | stable    | beta       |           |            |
|           | SVD DeepCache               | stable       | beta       | stable    | beta       |           |            |

**Note: Enterprise Edition contains all the functionality in Community Edition.**
* stable: release for public usage, and has long-term support; 
* beta: release for professional usage, and has long-term support; 
* alpha: early release for expert usage, and is **under active development**; 



### Acceleration for production environment
#### PyTorch Module compilation
- [compilation with oneflow_compile](https://github.com/siliconflow/onediff/blob/main/onediff_diffusers_extensions/examples/text_to_image_sdxl.py)
#### Avoid compilation time for new input shape
- [Support Multi-resolution input](https://github.com/siliconflow/onediff/blob/main/onediff_diffusers_extensions/examples/text_to_image_sdxl.py)
#### Avoid compilation time for online serving
Compile and save the compiled result offline, then load it online for serving
- [Save and Load the compiled graph](https://github.com/siliconflow/onediff/blob/main/onediff_diffusers_extensions/examples/text_to_image_sdxl_save_load.py)
- [Change device of the compiled graph to do multi-process serving](https://github.com/siliconflow/onediff/blob/main/onediff_diffusers_extensions/examples/text_to_image_sdxl_mp_load.py)

### OneDiff Enterprise Edition
If you need **Enterprise-level Support** for your system or business, you can 
- subscribe Enterprise Edition online and get all support after the order: https://siliconflow.com/onediff.html
- or send an email to contact@siliconflow.com and tell us about your user case, deployment scale, and requirements.

OneDiff Enterprise Edition can be **subscripted for one month and one GPU** and the cost is low.

|                      | OneDiff Enterprise   | OneDiff Community |
| -------------------- | ------------------- | ----------- |
| Multiple Resolutions | Yes(No time cost for most of the cases)       | Yes(No time cost for most of the cases)           |
| More Extreme and Dedicated optimization(usually another 20~100% performance gain)         |   Yes         |                 |
| Technical Support for deployment    | High priority support       | Community           |
| Get the experimental technology/feature | Yes | |

### Roadmap
[OneDiff Development Roadmap](https://github.com/siliconflow/onediff/wiki#onediff-roadmap)

## Community and Support
- [Create an issue](https://github.com/siliconflow/onediff/issues)
- Chat in Discord: [![](https://dcbadge.vercel.app/api/server/RKJTjZMcPQ?style=plastic)](https://discord.gg/RKJTjZMcPQ)
- Email for Enterprise Edition or other business inquiries: contact@siliconflow.com

## Installation
### OS and GPU support
- Linux
  - If you want to use OneDiff on Windows, please use it under WSL.
- NVIDIA GPUs

### OneDiff Installation

#### 1. Install OneFlow
> **_NOTE:_** We have updated OneFlow a lot for OneDiff, so please install OneFlow by the links below.

- **CUDA 11.8**

  ```bash
  # For NA/EU users
  python3 -m pip install -U --pre oneflow -f https://github.com/siliconflow/oneflow_releases/releases/expanded_assets/community_cu118
  ```


  ```bash
  # For CN users
  python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/community/cu118
  ```


<details>
<summary> Click to get OneFlow packages for other CUDA versions. </summary>

- **CUDA 12.1**

  ```bash
  # For NA/EU users
  python3 -m pip install -U --pre oneflow -f https://github.com/siliconflow/oneflow_releases/releases/expanded_assets/community_cu121
  ```


  ```bash
  # For CN users
  python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/community/cu121
  ```


- **CUDA 12.2**

  ```bash
  # For NA/EU users
  python3 -m pip install -U --pre oneflow -f https://github.com/siliconflow/oneflow_releases/releases/expanded_assets/community_cu122
  ```

  ```bash
  # For CN users
  python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/community/cu122
  ```



</details>


#### 2. Install torch and diffusers
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

```
python3 -m pip install huggingface_hub
 ~/.local/bin/huggingface-cli login
```

## Release

- run examples to check it works

  ```bash
  cd onediff_diffusers_extensions
  python3 examples/text_to_image.py
  ```

- bump version in these files:

  ```
  .github/workflows/pub.yml
  src/onediff/__init__.py
  ```

- install build package
  ```bash
  python3 -m pip install build
  ```

- build wheel

  ```bash
  rm -rf dist
  python3 -m build
  ```

- upload to pypi

  ```bash
  twine upload dist/*
  ```
