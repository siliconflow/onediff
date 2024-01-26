# OneDiff Diffusers Extensions

OneDiff diffusers extensions include multiple popular accelerated versions of the AIGC algorithm, such as DeepCache, which you would have a hard time finding elsewhere.

- [Install and Setup](#install-and-setup)
- [DeepCache Speedup](#deepcache-speedup)
    - [Stable Diffusion XL](#run-stable-diffusion-xl-with-onediff-diffusers-extensions)
    - [Stable Diffuison 1.5](#run-stable-diffusion-15-with-onediff-diffusers-extensions)
- [Contact](#contact)

## Install and setup

1. Follow the steps [here](https://github.com/siliconflow/onediff?tab=readme-ov-file#install-from-source) to install onediff. 

2. Install diffusers_extensions by following these steps

    ```
    git clone https://github.com/siliconflow/onediff.git
    cd onediff_diffusers_extensions && python3 -m pip install -e .
    ```

## DeepCache speedup

### Run Stable Diffusion XL with OneDiff diffusers extensions

```python
import torch

from onediff.infer_compiler import oneflow_compile
from diffusers_extensions.deep_cache import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.to("cuda")

pipe.unet = oneflow_compile(pipe.unet)
pipe.fast_unet = oneflow_compile(pipe.fast_unet)
pipe.vae = oneflow_compile(pipe.vae)

prompt = "A photo of a cat. Focus light and create sharp, defined edges."
# Warmup
for i in range(1):
    deepcache_output = pipe(
        prompt, 
        cache_interval=3, cache_layer_id=0, cache_block_id=0,
        output_type='pil'
    ).images[0]

deepcache_output = pipe(
    prompt, 
    cache_interval=3, cache_layer_id=0, cache_block_id=0,
    output_type='pil'
).images[0]
```

### Run Stable Diffusion 1.5 with OneDiff diffusers extensions

```python
import torch

from onediff.infer_compiler import oneflow_compile
from diffusers_extensions.deep_cache import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.to("cuda")

pipe.unet = oneflow_compile(pipe.unet)
pipe.fast_unet = oneflow_compile(pipe.fast_unet)
pipe.vae = oneflow_compile(pipe.vae)

prompt = "a photo of an astronaut on a moon"
# Warmup
for i in range(1):
    deepcache_output = pipe(
        prompt, 
        cache_interval=3, cache_layer_id=0, cache_block_id=0,
        output_type='pil'
    ).images[0]

deepcache_output = pipe(
    prompt, 
    cache_interval=3, cache_layer_id=0, cache_block_id=0,
    output_type='pil'
).images[0]
```

### Quantization

**Note**: Quantization feature is only supported by **OneDiff Enterprise**.

OneDiff Enterprise offers a quantization method that reduces memory usage, increases speed, and maintains quality without any loss.

If you possess a OneDiff Enterprise license key, you can access instructions on OneDiff quantization and related models by visiting [Hugginface/siliconflow](https://huggingface.co/siliconflow). Alternatively, you can [contact](#contact) us to inquire about purchasing the OneDiff Enterprise license.

## LoRA loading and switching speed up

OneDiff provides a faster implementation of loading LoRA, by invoking `diffusers_extensions.utils.lora.load_and_fuse_lora` you can load and fuse LoRA to pipeline.

```python
import torch
from diffusers import DiffusionPipeline
from onediff.infer_compiler import oneflow_compile
from diffusers_extensions.utils.lora import load_and_fuse_lora, unfuse_lora

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID, variant="fp16", torch_dtype=torch.float16
).to("cuda")

LORA_MODEL_ID = "hf-internal-testing/sdxl-1.0-lora"
LORA_FILENAME = "sd_xl_offset_example-lora_1.0.safetensors"

pipe.unet = oneflow_compile(pipe.unet)

# use onediff load_and_fuse_lora
load_and_fuse_lora(pipe, LORA_MODEL_ID, weight_name=LORA_FILENAME, lora_scale=1.0)
images_fusion = pipe(
    "masterpiece, best quality, mountain",
    height=1024,
    width=1024,
    num_inference_steps=30,
).images[0]
images_fusion.save("test_sdxl_lora.png")
```

We compared different methods of loading LoRA. The comparison of loading LoRA once is as shown in the table below.

| Method                           | Speed | Inference speed | LoRA loading speed    |
|----------------------------------|-------|------------------|-----------------------|
| load_lora_weight                 | 1.10s | low              | high                  |
| load_lora_weight + fuse_lora     | 1.38s | high             | low                   |
| onediff load_and_fuse_lora       | 0.56s | **high**         | **high**              |

If you want to unload LoRA and then load a new LoRA, you only need to call `load_and_fuse_lora` again. There is no need to manually call `unfuse_lora`, cause it will be called implicitly in `load_and_fuse_lora`. You can also manually call `unfuse_lora` to restore the model's weights.

## Contact

For users of OneDiff Community, please visit [GitHub Issues](https://github.com/siliconflow/onediff/issues) for bug reports and feature requests.

For users of OneDiff Enterprise, you can contact contact@siliconflow.com for commercial support.

Feel free to join our [Discord](https://discord.gg/RKJTjZMcPQ) community for discussions and to receive the latest updates.