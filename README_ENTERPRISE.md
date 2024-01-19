# OneDiff Enterprise

<p align="center">
<img src="imgs/onediff_logo.png" height="100">
</p>

OneDiff Enterprise offers a quantization method that reduces memory usage, increases speed, and maintains quality without any loss.

**Note**: Before proceeding with this document, please ensure you are familiar with the OneDiff Community features by referring to the [OneDiff Community README](./README.md).

- [Get the license key](#get-the-license-key)
- [Install OneDiff Enterprise](#install-onediff-enterprise)
- [ComfyUI with OneDiff Enterprise](#comfyui-with-onediff-enterprise)
    - [Accessing ComfyUI Models](#accessing-comfyui-models)
    - [Workflow](#workflow)
- [Diffusers with OneDiff Enterprise](#diffusers-with-onediff-enterprise)
    - [Accessing Diffusers Models](#accessing-diffusers-models)
    - [Run](#run)


## Get the license key

Purchase license key from [SiliconFlow website](https://www.siliconflow.com/onediff.html) or contact contact@siliconflow.com if you encounter any issues.

Alternatively, you can [contact](#contact) us to inquire about purchasing the OneDiff Enterprise license.

## Install OneDiff Enterprise

**CUDA 11.8**

```bash
python3 -m pip install onediff-quant -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/onediff-quant/cu118
```

**CUDA 12.1**

```bash
python3 -m pip install onediff-quant -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/onediff-quant/cu121
```

**CUDA 12.2**

```bash
python3 -m pip install onediff-quant -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/onediff-quant/cu122
```

## ComfyUI with OneDiff Enterprise

Ensure that you have installed [OneDiff ComfyUI Nodes](onediff_comfy_nodes/README.md#setup-enterprise-edition) and follow the instructions below.

### Accessing ComfyUI Models

To download the necessary models, please visit the [siliconflow/sdxl-base-1.0-onediff-comfy-enterprise-v1](https://huggingface.co/siliconflow/sdxl-base-1.0-onediff-comfy-enterprise-v1/tree/main) and the [siliconflow/stable-diffusion-v1-5-onediff-enterprise-v1](https://huggingface.co/siliconflow/stable-diffusion-v1-5-onediff-enterprise-v1/tree/main) on HuggingFace.

Place the `*.pt` files from the HuggingFace repositories into the `ComfyUI/models/onediff_quant` subfolder. If the `onediff_quant` folder does not exist, please create it.

### Workflow

#### SDXL

#### SDXL + DeepCache


## Diffusers with OneDiff Enterprise

### Accessing Diffusers Models

To download the necessary models, please visit the [siliconflow/sdxl-base-1.0-onediff-enterprise-v2](https://huggingface.co/siliconflow/sdxl-base-1.0-onediff-enterprise-v2/tree/main) on HuggingFace.

### Run

#### SDXL

Run [text_to_image_sdxl_enterprise.py](examples/text_to_image_sdxl_enterprise.py) by command:

```bash
python text_to_image_sdxl_enterprise.py --model $model_path --saved_image output_sdxl.png
```

Type `python3 text_to_image_sdxl_enterprise.py -h` for more options.

#### SDXL + DeepCache

Ensure that you have installed [OneDiff Diffusers Extensions](onediff_diffusers_extensions/README.md#install-and-setup) and then run [text_to_image_deep_cache_sdxl_enterprise.py](examples/text_to_image_deep_cache_sdxl_enterprise.py) by command:

```bash
python text_to_image_deep_cache_sdxl_enterprise.py --model $model_path --saved_image output_deepcache.png
```


## Contact

For users of OneDiff Community, please visit [GitHub Issues](https://github.com/siliconflow/onediff/issues) for bug reports and feature requests.

For users of OneDiff Enterprise, you can contact contact@siliconflow.com for commercial support.

Feel free to join our [Discord](https://discord.gg/RKJTjZMcPQ) community for discussions and to receive the latest updates.
