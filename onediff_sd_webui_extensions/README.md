# Stable-Diffusion-WebUI-OneDiff

- Performance of Community Edition
- [Installation Guide](#installation-guide)
- [Extensions Usage](#extensions-usage)

## Performance of Community Edition

Updated on DEC 13, 2023. Device: RTX 3090. Resolution: 1024x1024

torch(Baseline) | onediff(Optimized) | Percentage improvement
2.97it/s | 4.45it/s | 50%

## Installation Guide

It is recommended to create a Python virtual environment in advance. For example `conda create -n sd-webui python=3.10`.

```bash
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
git clone https://github.com/Oneflow-Inc/onediff.git
cd stable-diffusion-webui && git checkout 4afaaf8  # The tested git commit id is 4afaaf8.
cp -r ../onediff/onediff_sd_webui_extensions stable-diffusion-webui/extensions/

# Install all of stable-diffusion-webui's dependencies.
venv_dir=- bash webui.sh --port=8080

# Exit webui server and upgrade some of the components that conflict with onediff.
cd repositories/generative-models && git checkout 9d759324 && cd -
pip install -U einops==0.7.0
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

To enable OneDiff extension acceleration, select `onediff_diffusion_model` in Script and click the `Generate` button.

![onediff_script](images/onediff_script.jpg)
