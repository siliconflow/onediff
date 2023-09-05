# Stable Diffusion XL 1.0 OneFlow

## Performance Comparison
Timings for 30 steps at 1024x1024
|                         | Baseline | Optimized (oneflow.compile) | Percentage improvement |
| ----------------------- | -------- | --------------------------- | ---------------------- |
| NVIDIA GeForce RTX 3090 | 8433 ms  | 5161 ms                     | ~38.7%                 |
| A100 40G PCIE           | 4435 ms  | 3295 ms                     | ~25.7%                 |

## 🛠️ New Features 

- Convert PyTorch models like UNet to OneFlow static graph in one function
- Enable multi-resolution input with same compiled model 
- Save and load compiled static graph
  
## Installation Guide 🚀 NEW 

### Prerequisites

Before you begin, ensure that you meet the following prerequisites:

- NVIDIA driver with CUDA 11.7 support.
- Python environment with necessary packages, including `OneFlow>=0.9.1`.

### Installation

#### Clone the Repository and Install Requirements

1. Clone the repository and install the required packages from `requirements.txt` in a Python 3.10 environment, including `OneFlow` version 0.9.1 or higher:

    ```shell
    git clone https://github.com/Oneflow-Inc/diffusers.git
    cd diffusers 
    pip install .
    ```

#### Install OneFlow

- (Optional) If you are located in China, you can speed up the download by setting up a pip mirror:

    ```bash
    python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    ```

- Install `OneFlow`:

    ```bash
    python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu117
    ```

#### Install PyTorch

You can install PyTorch using either Conda or Pip:

- Using Conda (Recommended):

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```


- Using Pip:
```bash 
pip3 install torch torchvision torchaudio
```




## Demo

<details open>
<summary> demo.py </summary>

```python
import torch
# import oneflow as flow
from pathlib import Path
from onediff.infer_compiler import oneflow_compile
from diffusers import StableDiffusionXLPipeline


# Configuration
pretrained_model_name_or_path = "/share_nfs/hf_models/stable-diffusion-xl-base-1.0" # Please replace with your model path
prompt = "A little white cat playing on the sea floor, the sun shining in, swimming some beautiful Colorful goldfish, bubbles，by Yang J, pixiv contest winner, furry art, falling star on the background, bubbly underwater scenery, the cutest kitten ever, beautiful avatar pictures"
saved_image = "sdxl-base-out.png"
file_name = "unet_compiled"

pipe = StableDiffusionXLPipeline.from_pretrained(
    pretrained_model_name_or_path,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.to("cuda")

pipe.unet = oneflow_compile(pipe.unet)
if Path(file_name).exists():
    print(f"Loading the compiled graph from {file_name}...")
    pipe.unet._graph_load(file_name)


image = pipe(prompt).images[0]
print(f"Saved the image to {saved_image}.")
image.save(saved_image)

if not Path(file_name).exists():
    pipe.unet._graph_save(file_name)
    print(f"Saved the compiled graph to {file_name}.")

```
</details>

 
To test compiling and saving the UNet model, run the following command:

```shell 
python demo.py && python demo.py 
# First: Compile and save the UNet model
# Second: Load the compiled UNet model
```


<details open>
<summary> Output: sdxl-base-out.png </summary>

![sdxl-base-out.png](https://github.com/Oneflow-Inc/OneTeam/assets/109639975/a9f7bfe8-0f84-43ea-855b-efb4e0fde96f)
</details>


### related examples

<a href="https://github.com/Oneflow-Inc/diffusers/blob/refactor-backend/examples/text_to_image_sdxl_fp16.py" target="_new">Example 1: <code>text_to_image_sdxl_fp16.py</code></a>

<a href="https://github.com/Oneflow-Inc/diffusers/blob/refactor-backend/examples/text_to_image_sdxl_fp16_with_oneflow_compile.py" target="_new">Example 2: <code>text_to_image_sdxl_fp16_with_oneflow_compile.py</code></a>




