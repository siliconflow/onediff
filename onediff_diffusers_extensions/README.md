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


OneDiff provides a more efficient implementation of loading LoRA, by invoking `load_and_fuse_lora` you can load and fuse LoRA to pipeline, and by invoking `unfuse_lora` you can restore the weight of base model.

### API
`onediffx.utils.lora.load_and_fuse_lora(pipeline: LoraLoaderMixin, pretrained_model_name_or_path_or_dict: Union[str, Path, Dict[str, torch.Tensor]], adapter_name: Optional[str] = None, *, lora_scale: float = 1.0, offload_device="cpu", offload_weight="lora", use_cache=False, **kwargs)`:
- pipeline (`LoraLoaderMixin`): The pipeline that will load and fuse LoRA weight.

- pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):  Can be either:

    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on the Hub.

    - A path to a *directory* containing the model weights saved with [ModelMixin.save_pretrained()](https://huggingface.co/docs/diffusers/v0.25.1/en/api/models/overview#diffusers.ModelMixin.save_pretrained).

    - A [torch state dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

- adapter_name(`str`, *optional*): Adapter name to be used for referencing the loaded adapter model. If not specified, it will use `default_{i}` where i is the total number of adapters being loaded. **Not supported now**.

- lora_scale (`float`, defaults to 1.0): Controls how much to influence the outputs with the LoRA parameters.

- offload_device (`str`, must be one of "cpu" and "cuda"): The device to offload the weight of LoRA or model

- offload_weight (`str`, must be one of "lora" and "weight"): The weight type to offload. If set to "lora", the weight of LoRA will be offloaded to `offload_device`, and if set to "weight", the weight of Linear or Conv2d will be offloaded.

- use_cache (`bool`, optional): Whether to save LoRA to cache. If set to True, loaded LoRA will be cached in memory.

- kwargs(`dict`, *optional*) — See [lora_state_dict()](https://huggingface.co/docs/diffusers/v0.25.1/en/api/loaders/lora#diffusers.loaders.LoraLoaderMixin.lora_state_dict)

  

`onediffx.utils.lora.unfuse_lora(pipeline: LoraLoaderMixin) -> None`:

- pipeline (`LoraLoaderMixin`): The pipeline that will unfuse LoRA weight.

### Example

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
  
  # unload LoRA weights and restore base model
  unfuse_lora(pipe)
```

### Benchmark

We choose 5 LoRAs to profile loading and switching speed of 3 different APIs

1. `load_lora_weight`, which has high loading performance but low inference performance

2. `load_lora_weight + fuse_lora`, which has high inference performance but low loading performance

3. `onediffx.utils.lora.load_and_fuse_lora`, which has high loading performance and high inference performance


The results are shown below

| LoRA name | size | load_lora_weight | load_lora_weight + fuse_lora | **onediffx load_and_fuse_lora** | unet cnt | te1 cnt | te2 cnt | src link |
|---|---|---|---|---|---|---|---|---|
| SDXL-Emoji-Lora-r4.safetensors | 28M | 1.69 s | 2.34 s | **0.78 s** | 2166 | 216 | 576 | https://novita.ai/model/SDXL-Emoji-Lora-r4_160282 |
| sdxl_metal_lora.safetensors | 23M | 0.97 s | 1.73 s | **0.19 s** | 1120 | 0 | 0 | |
| simple_drawing_xl_b1-000012.safetensors | 55M | 1.67 s | 2.57 s | **0.77 s** | 2166 | 216 | 576 | https://civitai.com/models/177820/sdxl-simple-drawing |
| texta.safetensors | 270M | 1.72 s | 2.86 s | **0.97 s** | 2364 | 0 | 0 | https://civitai.com/models/221240/texta-generate-text-with-sdxl |
| watercolor_v1_sdxl_lora.safetensors | 12M | 1.54 s | 2.01 s | **0.35 s** | 1680 | 0 | 0 | |


## Contact

For users of OneDiff Community, please visit [GitHub Issues](https://github.com/siliconflow/onediff/issues) for bug reports and feature requests.

For users of OneDiff Enterprise, you can contact contact@siliconflow.com for commercial support.

Feel free to join our [Discord](https://discord.gg/RKJTjZMcPQ) community for discussions and to receive the latest updates.