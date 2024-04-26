<p align="center">
<img src="imgs/onediff_logo.png" height="100">
</p>

## OneDiff on wsl2
## Installation
### OneDiff Installation on wsl2
This tutorial provides a way to run OneDiff on a WSL2 system.

#### 1. Install wsl2
If you have nVidia GPU driver installed locally, then you can use it directly in wsl.If not, please install the graphics card driver first.

You can follow this URL to install wsl2
https://learn.microsoft.com/en-us/windows/wsl/install-manual.

>**_NOTE:_**If you run into such a problem `` Prompt when WSL2 starts: The referenced object type does not support the attempted operation``Here are some things you can try\
https://github.com/microsoft/WSL/issues/4177#issuecomment-1429113508 \

After preparing the system, you can use anaconda and miniconda to manage the environment, use python 3.10 to create the environment, and install torch 2.22 in the environment after finishing.

#### 2. Install OneFlow
> **_NOTE:_** 1.If you are a CN user, please update the Ubuntu system source <br>2. We have updated OneFlow frequently for OneDiff, so please install OneFlow by the links below.

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
