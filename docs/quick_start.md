
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


def compile_and_save_unet(model, file_name):
    model.unet = oneflow_compile(model.unet)
    print("unet is compiled to oneflow.")
    pipe(prompt).images[0]
    print(f"Saving the compiled unet model to {file_name}...")
    model.unet._graph_save(file_name)


def load_compiled_unet(model, file_name):
    model.unet = oneflow_compile(model.unet)
    print("unet is compiled to oneflow.")
    model.unet._graph_load(file_name)
    print(f"Loaded the compiled unet model from {file_name}.")


pipe = StableDiffusionXLPipeline.from_pretrained(
    pretrained_model_name_or_path,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.to("cuda")

if Path(file_name).exists():
    load_compiled_unet(pipe, file_name)
    image = pipe(prompt).images[0]
    image.save(saved_image)
else:
    compile_and_save_unet(pipe, file_name)

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


![Alt text](image.png)
</details>



**Note:**

To enable CUDA graph for OneFlow, you can use the following flags for acceleration:

```shell
export ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH=1 && export ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1
```
When using Stable Diffusion XL (SDXL) on an A100 GPU with a resolution of 1024x1024 and 30 inference steps, you can expect a speedup of approximately 0.1 iterations per second. However, the performance improvement may not be significant on an RTX 3090. For smaller resolutions, such as 128x128, you may achieve a doubling of acceleration.

### related examples

<a href="https://github.com/Oneflow-Inc/diffusers/blob/refactor-backend/examples/text_to_image_sdxl_fp16.py" target="_new">Example 1: <code>text_to_image_sdxl_fp16.py</code></a>

<a href="https://github.com/Oneflow-Inc/diffusers/blob/refactor-backend/examples/text_to_image_sdxl_fp16_with_oneflow_compile.py" target="_new">Example 2: <code>text_to_image_sdxl_fp16_with_oneflow_compile.py</code></a>


