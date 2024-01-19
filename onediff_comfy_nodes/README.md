# OneDiff ComfyUI Nodes

<p align="center">
<img src="../imgs/onediff_logo.png" height="100">
</p>

Performance of Community Edition

Updated on DEC 7, 2023. Device: RTX 3090

| SDXL1.0-base (1024x1024)                                       | torch(Baseline) | onediff(Optimized) | Percentage improvement |
| -------------------------------------------------------------- | --------------- | ------------------ | ---------------------- |
| [Stable Diffusion workflow(UNet)](workflows/model-speedup.png) | 4.08it/s        | 6.70it/s           | 64.2 %                 |
| [LoRA workflow](workflows/model-speedup-lora.png)              | 4.05it/s        | 6.69it/s           | 65.1 %                 |

## Documentation

- [Installation Guide](#installation-guide)
- [Basical Nodes Usage](#basical-nodes-usage)
  - [OneDiff LoadCheckpoint ](#load-checkpoint---onediff)
  - [Quantization](#quantization)
- [OneDiff Community Examples](#onediff-community-examples)
  - [LoRA](#lora)
  - [ControlNet](#controlnet)
  - [SVD](#svd)
  - [DeepCache](#deepcache)
- [Contact](#contact)


### Installation Guide

Please install and set up [ComfyUI](https://github.com/comfyanonymous/ComfyUI) first, and then:

#### Setup Community Edition

<details close>
<summary>Setup Community Edition</summary>

1. Install OneFlow Community
  * Install OneFlow Community(CUDA 11.x)

    ```bash
    pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/community/cu118
    ```

  * Install OneFlow Community(CUDA 12.x)

    ```bash
    pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/community/cu121
    ```
2. Install OneDiff
    ```bash
    git clone https://github.com/siliconflow/onediff.git
    cd onediff && pip install -e .
    ```

3. Install onediff_comfy_nodes for ComfyUI

    ```bash
    cd onediff
    cp -r onediff_comfy_nodes path/to/ComfyUI/custom_nodes/
    ```

</details>

#### Setup Enterprise Edition

<summary>Setup Enterprise Edition</summary>

1. [Install OneDiff Enterprise](../README_ENTERPRISE.md#install-onediff-enterprise)

2. Install onediff_comfy_nodes for ComfyUI
    ```bash
    git clone https://github.com/siliconflow/onediff.git
    cd onediff 
    cp -r onediff_comfy_nodes path/to/ComfyUI/custom_nodes/
    ```

</details>


### Basical Nodes Usage

**Note** All the images in this section can be loaded directly into ComfyUI. You can load them in ComfyUI to get the full workflow.

#### Load Checkpoint - OneDiff

"Load Checkpoint - OneDiff" is the optimized version of "LoadCheckpoint", designed to accelerate the inference speed without any awareness required. It maintains the same input and output as the original node.

![](workflows/model-speedup.png)


The "Load Checkpoint - OneDiff" node  set `vae_speedup` :  `enable` to enable VAE acceleration.


### Quantization

**Note**: Quantization feature is only supported by **OneDiff Enterprise**.

OneDiff Enterprise offers a quantization method that reduces memory usage, increases speed, and maintains quality without any loss.

If you possess a OneDiff Enterprise license key, you can access instructions on OneDiff quantization and related models by visiting [Hugginface/siliconflow](https://huggingface.co/siliconflow). Alternatively, you can [contact](#contact) us to inquire about purchasing the OneDiff Enterprise license.

![](workflows/onediff_quant_base.png)


## OneDiff Community Examples 

### LoRA                  

This example demonstrates how to utilize LoRAs. You have the flexibility to modify the LoRA models or adjust their strength without the need for recompilation.

[Lora Speedup](workflows/model-speedup-lora.png)

### ControlNet

While there is an example demonstrating OpenPose ControlNet, it's important to note that OneDiff seamlessly supports a wide range of ControlNet types, including depth mapping, canny, and more.

[ControlNet Speedup](workflows/model-speedup-controlnet.png)

### SVD

This example illustrates how OneDiff can be used to enhance the performance of a video model, specifically in the context of text-to-video generation using SVD.

[SVD Speedup](workflows/text-to-video-speedup.png)

### DeepCache

DeepCache is an innovative algorithm that substantially boosts the speed of diffusion models, achieving an approximate 2x improvement. When used in conjunction with OneDiff, it further accelerates the diffusion model to approximately 3x.

Here are the example of applying DeepCache to SD and SVD models.

[Module DeepCache SpeedUp on SD](workflows/deep-cache.png)

[Module DeepCache SpeedUp on SVD](workflows/svd-deepcache.png)

[Module DeepCache SpeedUp on LoRA](workflows/lora_deepcache/README.md) 

## Contact

For users of OneDiff Community, please visit [GitHub Issues](https://github.com/siliconflow/onediff/issues) for bug reports and feature requests.

For users of OneDiff Enterprise, you can contact contact@siliconflow.com for commercial support.

Feel free to join our [Discord](https://discord.gg/RKJTjZMcPQ) community for discussions and to receive the latest updates.
