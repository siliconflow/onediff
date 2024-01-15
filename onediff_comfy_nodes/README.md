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

### Installation Guide

Please install and set up [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

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



<details close>
<summary>Setup Enterprise Edition</summary>

1. Install OneFlow Enterprise
    ```bash
      python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/main/cu121
    ```

2. Get license key from [SiliconFlow website](https://www.siliconflow.com/onediff.html)

3. Set up the key

  ```bash
  export SILICON_ONEDIFF_LICENSE_KEY=YOUR_LICENSE_KEY
  ```


2. Install OneDiff and OneDiff Quant
    ```bash
    python3 -m pip install onediff-quant -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/onediff-quant && \
    git clone https://github.com/siliconflow/onediff.git  && \
    cd onediff && pip install -e . && cd ..
    ```
3. Install onediff_comfy_nodes for ComfyUI
    ```bash
    cd onediff 
    cp -r onediff_comfy_nodes path/to/ComfyUI/custom_nodes/
    ```

</details>


### Basical Nodes Usage

**Note** All the images in this section can be loaded directly into ComfyUI. 

You can Load these images in ComfyUI to get the full workflow.

#### Load Checkpoint - OneDiff

The "Load Checkpoint - OneDiff" node  is optimized for OneDiff. 

It can be used to load checkpoints and accelerate the model.

![](workflows/model-speedup.png)


The "Load Checkpoint - OneDiff" node  set `vae_speedup` :  `enable` to enable VAE acceleration.


### Quantization

**Note: Quantization feature is only supported in OneDiff Enterprise.**


![](workflows/onediff_quant_base.png)

The compilation result of the quantized model can also be saved as a graph and loaded when needed.

### Image Distinction Scanner

The "Image Distinction Scanner" node is used to compare the differences between two images and visualize the resulting variances.

![](workflows/image-distinction-scanner.png)

## OneDiff Community Examples 

### LoRA                  

This example shows you how to use Loras. You can change the LoRA models or adjust their strength without needing to recompile.

[Lora Speedup](workflows/model-speedup-lora.png)

### ControlNet

There is an example demonstrating openpose controlnet while OneDiff seamlessly supports a wide range of controlnet types, including depth mapping, canny, and more.

[ControlNet Speedup](workflows/model-speedup-controlnet.png)

### SVD

This example demonstrates the utilization of OneDiff to enhance the performance of a video model (text to video by SVD)

[SVD Speedup](workflows/text-to-video-speedup.png)

### DeepCache

DeepCache is an innovative algorithm designed to significantly enhance the speed of diffusion models by approximately 2x. When combined with OneDiff, it further accelerates the Diffusion model by around 3x.

Here are the example of applying DeepCache to SD and SVD models.

[Module DeepCache SpeedUp on SD](workflows/deep-cache.png)

[Module DeepCache SpeedUp on SVD](workflows/svd-deepcache.png)

[Module DeepCache SpeedUp on LoRA](workflows/lora_deepcache/README.md) 

## <div align="center">Contact</div>

For OneDiff bug reports and feature requests please visit [GitHub Issues](https://github.com/siliconflow/onediff/issues), and join our [Discord](https://discord.gg/RKJTjZMcPQ) community for questions and discussions!




