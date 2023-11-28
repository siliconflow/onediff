[![PyPI version](https://badge.fury.io/py/onediff.svg)](https://badge.fury.io/py/onediff)
[![Docker image build](https://github.com/Oneflow-Inc/diffusers/actions/workflows/sd.yml/badge.svg)](https://github.com/Oneflow-Inc/diffusers/actions/workflows/sd.yml)

# OneDiff

A **drop-in acceleration lib** for diffusers and other **diffusion models**.

## Performance of OneDiff Community Edition 

Updated on Nov 6, 2023.

|     Device     | SD1.5 (512x512) | SD2.1 (512x512) | SDXL1.0-base（1024x1024） |
| -------------- | --------------- | --------------- | ------------------------- |
| RTX 3090       | 42.38it/s       | 42.33it/s       | 6.66it/s                  |
| RTX 4090       | 74.71it/s       | 73.57it/s       | 13.57it/s                 |
| A100-PCIE-40GB | 54.4it/s        | 54.06it/s       | 10.22it/s                 |
| A100-SXM4-80GB | 59.68it/s       | 61.91it/s       | 11.80it/s                 |

> **_NOTE:_** OneDiff Enterprise Edition delivers even higher performance and second-to-none deployment flexibility.

## Features
- Acceleration for SDXL, SD2.1 and so on
  - [SDXL](https://github.com/Oneflow-Inc/diffusers/blob/main/examples/text_to_image_sdxl.py) 
  - [SD2.1](https://github.com/Oneflow-Inc/diffusers/blob/main/examples/text_to_image.py)
  - [LoRA](https://github.com/Oneflow-Inc/diffusers/blob/main/examples/text_to_image_sdxl_lora.py)
  - [LCM](https://github.com/Oneflow-Inc/diffusers/blob/main/examples/text_to_image_lcm.py) and [LCM LoRA](https://github.com/Oneflow-Inc/diffusers/blob/main/examples/text_to_image_lcm_lora_sdxl.py)
- Compile PyTorch models (like UNet) with one function
  - [oneflow_compile](https://github.com/Oneflow-Inc/diffusers/blob/a38c5ea475c07b4527981ec5723ccac083ed0a9c/examples/text_to_image_sdxl.py#L53)
- Enable multi-resolution input with the same compiled model
  - [The Multi graph way](https://github.com/Oneflow-Inc/diffusers/blob/a38c5ea475c07b4527981ec5723ccac083ed0a9c/examples/text_to_image_sdxl_save_load.py#L65)
- Save/load/change device of the compiled graph
  - [Save and Load](https://github.com/Oneflow-Inc/diffusers/blob/main/examples/text_to_image_sdxl_save_load.py)
  - [Change device to do multi-process serving](https://github.com/Oneflow-Inc/diffusers/blob/main/examples/text_to_image_sdxl_mp_load.py)

## Business inquiry on OneDiff Enterprise Edition

If you need **LoRA**, [**ComfyUI**](https://github.com/Oneflow-Inc/diffusers/wiki/Run-ComfyUI-with-OneDiff), **quant** support or any other more advanced features, please send an email to caishenghang@oneflow.org . Tell us about your **use case, deployment scale and requirements**! 
|                      | OneDiff Community   | OneDiff Enterprise|
| -------------------- | ------------------- | ----------- |
| diffusers            | Yes                 | Yes         |
| UNet/VAE/ControlNet Compilation | Yes      | Yes         |
| LoRA                 | Limited             | Yes         |
| LCM                  | Limited             | Yes         |
| Multiple Resolutions | Limited             | Yes         |
| Technical Support    | Community           | Yes         |
| ComfyUI              |                     | Yes         |
| Quantization         |                     | Yes         |
| Source Code Access   |                     | Yes         |

## Install with pip or Using in Docker
### Install from source

1. Install OneFlow(For CUDA 11.8)
```
python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/master_open_source/cu118
```
OneFlow packages for other CUDA versioins:
<details>
CUDA 12.1

```bash
python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/master_open_source/cu121
```

CUDA 12.2

```bash
python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/master_open_source/cu122
```

</details>


2. Install torch and diffusers
```
python3 -m pip install "torch" "transformers==4.27.1" "diffusers[torch]==0.19.3"
```

3. Install OneDiff
```
git clone https://github.com/Oneflow-Inc/diffusers.git onediff
cd onediff && python3 -m pip install -e .
```

4. (Optional)Login huggingface-cli

```
python3 -m pip install huggingface_hub
 ~/.local/bin/huggingface-cli login
```

### Docker
```bash
docker pull oneflowinc/onediff:20231106
```


### Run examples

```
python3 examples/text_to_image.py
```

### Release

- run examples to check it works

  ```bash
  python3 examples/text_to_image.py --model_id=...
  python3 examples/text_to_image_sdxl.py --base ...
  bash examples/unet_save_and_load.sh --model_id=...
  ```

- bump version in these files:

  ```
  setup.py
  src/onediff/__init__.py
  ```

- build wheel

  ```
  rm -rf dist
  python3 setup.py bdist_wheel
  ```

- upload to pypi

  ```bash
  twine upload dist/*
  ```

## More about OneFlow

OneFlow's main [repo](https://github.com/Oneflow-Inc/oneflow)
