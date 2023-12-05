# Stable-Diffusion-WebUI-OneDiff

- [Installation Guide](#installation-guide)
- [Extensions Usage](#extensions-usage)

## Installation Guide

1. Install and set up Stable Diffusion web UI based on [this commit snapshot](https://github.com/AUTOMATIC1111/stable-diffusion-webui/tree/4afaaf8a020c1df457bcf7250cb1c7f609699fa7).

Perform the following [manual installation](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs#manual-installation) process:
```bash
# conda create -n sd-webui python=3.10
# conda activate sd-webui

python -m pip install tb-nightly

cd stable-diffusion-webui
mkdir repositories
git clone https://github.com/salesforce/BLIP.git repositories/BLIP && cd repositories/BLIP && git checkout 3a29b741 && cd -
git clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer && cd repositories/CodeFormer && git checkout 8392d033 && cd -
    # pip install -r repositories/CodeFormer/requirements.txt --prefer-binary
git clone https://github.com/Stability-AI/generative-models.git repositories/generative-models && \
    cd repositories/generative-models && git checkout 9d759324 && cd - && \
    pip3 install -r repositories/generative-models/requirements/pt2.txt
git clone https://github.com/Stability-AI/stablediffusion.git repositories/stable-diffusion-stability-ai && cd repositories/stable-diffusion-stability-ai && git checkout cf1d67a6 && cd -
git clone https://github.com/CompVis/taming-transformers.git repositories/taming-transformers && cd repositories/taming-transformers && git checkout 3ba01b24 && cd -

pip install diffusers invisible-watermark --prefer-binary
pip install git+https://github.com/crowsonkb/k-diffusion.git --prefer-binary
# pip install git+https://github.com/TencentARC/GFPGAN.git --prefer-binary

pip install -r requirements.txt --prefer-binary
# dependency conflicts
pip install -U numpy torchdata --prefer-binary

# ImportError: cannot import name 'Doc' from 'typing_extensions'
pip install typing-extensions==4.8.0
# ImportError: cannot import name 'rank_zero_only' from 'pytorch_lightning.utilities.distributed'
python -m pip install pytorch-lightning==1.9.5
# AttributeError: __config__
pip install pydantic==1.10.13
```

2. Install OneFlow

```bash
pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/community/cu118
```

3. Intall onediff

```bash
git clone https://github.com/Oneflow-Inc/diffusers.git onediff
cd onediff && python3 -m pip install -e .
```

6. Copy files from onediff to stable-diffusion-webui `extensions` folder

```bash
cp -r onediff/onediff_sd_webui_extensions \
      stable-diffusion-webui/extensions
```

7. Copy model file

```bash
cp sd_xl_base_1.0.safetensors stable-diffusion-webui/models/Stable-diffusion/
```

To improve the speed of the first startup of the service, download [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) ahead of time and change the model path in [the code](https://github.com/Stability-AI/generative-models/blob/059d8e9cd9c55aea1ef2ece39abf605efb8b7cc9/sgm/modules/encoders/modules.py#L338). The example code is as follows:
```python
class FrozenCLIPEmbedder(AbstractEmbModel):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        # the path of openai/clip-vit-large-patch14
        version="/data/clip-vit-large-patch14",
```

## Run stable-diffusion-webui service

```bash
cd stable-diffusion-webui
python webui.py --port 8080
```

Accessing http://server:8080/ from a web browser.

## Extensions Usage

Type prompt in the text box, such as `a black dog`. Click the `Generate` button in the upper right corner to generate the image. As you can see in the image below:

![raw_webui](images/raw_webui.jpg)

To enable onediff plugin acceleration, select `onediff` in Script and click the `Generate` button.

![onediff_script](images/onediff_script.jpg)
