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
    cd diffusers_extensions && python3 -m pip install -e .
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

## Contact

For users of OneDiff Community, please visit [GitHub Issues](https://github.com/siliconflow/onediff/issues) for bug reports and feature requests.

For users of OneDiff Enterprise, you can contact contact@siliconflow.com for commercial support.

Feel free to join our [Discord](https://discord.gg/RKJTjZMcPQ) community for discussions and to receive the latest updates.