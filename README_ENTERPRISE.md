# OneDiff Enterprise

<p align="center">
<img src="imgs/onediff_logo.png" height="100">
</p>

OneDiff Enterprise offers a quantization method that reduces memory usage, increases speed, and maintains quality without any loss.

**Note**: Before proceeding with this document, please ensure you are familiar with the OneDiff Community features by referring to the [OneDiff Community README](./README.md).

- [Get the license key](#get-the-license-key)
- [Install OneDiff Enterprise](#install-onediff-enterprise)
    - [For NA/EU users](#for-naeu-users)
    - [For CN users](#for-cn-users)
- [ComfyUI with OneDiff Enterprise](#comfyui-with-onediff-enterprise)
    - [Accessing ComfyUI Models](#accessing-comfyui-models)
    - [Workflow](#workflow)
- [Diffusers with OneDiff Enterprise](#diffusers-with-onediff-enterprise)
    - [SDXL](#SDXL)
        - [Accessing Diffusers Models](#accessing-diffusers-models)
        - [Scripts](#scripts)
    - [SVD](#SVD)
        - [Accessing Diffusers Models](#accessing-diffusers-models)
        - [Scripts](#scripts)


## Get the license key

Purchase license key from [SiliconFlow website](https://www.siliconflow.com/onediff.html) or contact contact@siliconflow.com if you encounter any issues.

Alternatively, you can [contact](#contact) us to inquire about purchasing the OneDiff Enterprise license.

## Install OneDiff Enterprise

### For NA/EU users

**CUDA 11.8**

```bash
python3 -m pip install --pre oneflow -f https://github.com/siliconflow/oneflow_releases/releases/expanded_assets/enterprise_cu118 && \
python3 -m pip install --pre onediff-quant -f https://github.com/siliconflow/onediff_releases/releases/expanded_assets/enterprise && \
python3 -m pip install git+https://github.com/siliconflow/onediff.git@main#egg=onediff
```

**CUDA 12.1**

```bash
python3 -m pip install --pre oneflow -f https://github.com/siliconflow/oneflow_releases/releases/expanded_assets/enterprise_cu121 && \
python3 -m pip install --pre onediff-quant -f https://github.com/siliconflow/onediff_releases/releases/expanded_assets/enterprise && \
python3 -m pip install git+https://github.com/siliconflow/onediff.git@main#egg=onediff
```

**CUDA 12.2**

```bash
python3 -m pip install --pre oneflow -f https://github.com/siliconflow/oneflow_releases/releases/expanded_assets/enterprise_cu122 && \
python3 -m pip install --pre onediff-quant -f https://github.com/siliconflow/onediff_releases/releases/expanded_assets/enterprise && \
python3 -m pip install git+https://github.com/siliconflow/onediff.git@main#egg=onediff
```

### For CN users

**CUDA 11.8**

```bash
python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/main/cu118/ && \
python3 -m pip install --pre onediff-quant -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/onediff-quant/ && \
python3 -m pip install git+https://github.com/siliconflow/onediff.git@main#egg=onediff
```

**CUDA 12.1**

```bash
python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/main/cu121/ && \
python3 -m pip install --pre onediff-quant -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/onediff-quant/ && \
python3 -m pip install git+https://github.com/siliconflow/onediff.git@main#egg=onediff
```

**CUDA 12.2**

```bash
python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/main/cu122/ && \
python3 -m pip install --pre onediff-quant -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/onediff-quant/ && \
python3 -m pip install git+https://github.com/siliconflow/onediff.git@main#egg=onediff
```

## ComfyUI with OneDiff Enterprise

Ensure that you have installed [OneDiff ComfyUI Nodes](onediff_comfy_nodes/README.md#setup-enterprise-edition) and follow the instructions below.

### Accessing ComfyUI Models


To download the necessary models:


1. **SD 1.5**
   - For more information and to access the model, visit [Hugging Face - stable-diffusion-v1-5-onediff-enterprise-v1](https://huggingface.co/siliconflow/stable-diffusion-v1-5-onediff-comfy-enterprise-v1/tree/main).

2. **SDXL**
   - For details, visit [Hugging Face - sdxl-base-1.0-onediff-comfy-enterprise-v1](https://huggingface.co/siliconflow/sdxl-base-1.0-onediff-comfy-enterprise-v1/tree/main).

3. **SVD**
   - For details, visit [Hugging Face - stable-video-diffusion-xt-comfyui-deepcache-int8](https://huggingface.co/siliconflow/stable-video-diffusion-xt-comfyui-deepcache-int8).


**NOTE**: Place the `*.pt` files from the HuggingFace repositories into the `ComfyUI/models/onediff_quant` subfolder. If the `onediff_quant` folder does not exist, please create it.

### Workflow

Click the links below to view the workflow images, or load them directly into ComfyUI.

- [SD 1.5](https://huggingface.co/siliconflow/stable-diffusion-v1-5-onediff-enterprise-v1/blob/main/comfyui_screenshots/onediff_quant_advanced.png)
- [SDXL](https://huggingface.co/siliconflow/sdxl-base-1.0-onediff-comfy-enterprise-v1/blob/main/onediff_quant_base.png)
- [SDXL + DeepCache](https://huggingface.co/siliconflow/sdxl-base-1.0-onediff-comfy-enterprise-v1/blob/main/onediff_quant_deepcache.png)
- [SVD](https://huggingface.co/siliconflow/stable-video-diffusion-xt-comfyui-deepcache-int8/blob/main/svd-int8-workflow.png)
- [SVD + DeepCache](https://huggingface.co/siliconflow/stable-video-diffusion-xt-comfyui-deepcache-int8/blob/main/svd-int8-deepcache-workflow.png)

## Diffusers with OneDiff Enterprise

### SDXL

#### Accessing Diffusers Models

To download the necessary models, please visit the [siliconflow/sdxl-base-1.0-onediff-enterprise-v2](https://huggingface.co/siliconflow/sdxl-base-1.0-onediff-enterprise-v2/tree/main) on HuggingFace.

#### Scripts

Run [text_to_image_sdxl_enterprise.py](examples/text_to_image_sdxl_enterprise.py) by command:

```bash
python text_to_image_sdxl_enterprise.py --model $model_path --saved_image output_sdxl.png
```

Type `python3 text_to_image_sdxl_enterprise.py -h` for more options.

#### SDXL + DeepCache

Ensure that you have installed [OneDiffX](onediff_diffusers_extensions/README.md#install-and-setup) and then run [text_to_image_deep_cache_sdxl_enterprise.py](examples/text_to_image_deep_cache_sdxl_enterprise.py) by command:

```bash
python text_to_image_deep_cache_sdxl_enterprise.py --model $model_path --saved_image output_deepcache.png
```

### SVD

#### Accessing Diffusers Models

To download the necessary models, please visit the [siliconflow/stable-video-diffusion-img2vid-xt-deepcache-int8](https://huggingface.co/siliconflow/stable-video-diffusion-img2vid-xt-deepcache-int8) on HuggingFace.

#### Scripts

Run [image_to_video.py](benchmarks/image_to_video.py):

```bash
python3 benchmarks/image_to_video.py \     
  --model $model_path \    
  --input-image path/to/input_image.jpg \     
  --output-video path/to/output_image.mp4   
```

#### SVD + DeepCache

```bash
python3 benchmarks/image_to_video.py \     
  --model $model_path \     
  --deepcache \     
  --input-image path/to/input_image.jpg \     
  --output-video path/to/output_image.mp4 
```


## Contact

For users of OneDiff Community, please visit [GitHub Issues](https://github.com/siliconflow/onediff/issues) for bug reports and feature requests.

For users of OneDiff Enterprise, you can contact contact@siliconflow.com for commercial support.

Feel free to join our [Discord](https://discord.gg/RKJTjZMcPQ) community for discussions and to receive the latest updates.
