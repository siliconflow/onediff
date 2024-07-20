# OneDiffX (for HF diffusers)

OneDiffX is a OneDiff Extension for HF diffusers. It provides some acceleration utilities, such as DeepCache.

- [Install and Setup](#install-and-setup)
- [Compile, save and load pipeline](#compile-save-and-load-pipeline)
- Acceleration for state-of-the-art Models
  - [SDXL](https://github.com/siliconflow/onediff/blob/main/onediff_diffusers_extensions/examples/text_to_image_sdxl.py)
  - [SDXL Turbo](https://github.com/siliconflow/onediff/blob/main/onediff_diffusers_extensions/examples/text_to_image_sdxl_turbo.py)
  - [SD 1.5/2.1](https://github.com/siliconflow/onediff/blob/main/onediff_diffusers_extensions/examples/text_to_image.py)
  - [LoRA (and dynamic switching LoRA)](https://github.com/siliconflow/onediff/blob/main/onediff_diffusers_extensions/examples/text_to_image_sdxl_lora.py)
  - [ControlNet](https://github.com/siliconflow/onediff/blob/main/onediff_diffusers_extensions/examples/text_to_image_controlnet.py)
  - [LCM](https://github.com/siliconflow/onediff/blob/main/onediff_diffusers_extensions/examples/text_to_image_lcm.py) and [LCM LoRA](https://github.com/siliconflow/onediff/blob/main/onediff_diffusers_extensions/examples/text_to_image_lcm_lora_sdxl.py)
  - [Stable Video Diffusion](https://github.com/siliconflow/onediff/blob/main/onediff_diffusers_extensions/examples/image_to_video.py)
  - [DeepCache](https://github.com/siliconflow/onediff/blob/main/onediff_diffusers_extensions/examples/text_to_image_deep_cache_sdxl.py)
  - [InstantID](https://github.com/siliconflow/onediff/blob/main/benchmarks/instant_id.py)
- [DeepCache Speedup](#deepcache-speedup)
    - [Stable Diffusion XL](#run-stable-diffusion-xl-with-onediffx)
    - [Stable Diffusion 1.5](#run-stable-diffusion-15-with-onediffx)
- [Fast LoRA loading and switching](#fast-lora-loading-and-switching)
- [Quantization](#quantization)
- [Contact](#contact)

## Install and setup

1. Follow the steps [here](https://github.com/siliconflow/onediff?tab=readme-ov-file#install-from-source) to install onediff. 

2. Install onediffx by following these steps

    ```
    git clone https://github.com/siliconflow/onediff.git
    cd onediff_diffusers_extensions && python3 -m pip install -e .
    ```
## Compile, save and load pipeline
The complete example to test compile/save/load the pipeline: [pipe_compile_save_load.py](examples/pipe_compile_save_load.py).
### Compile diffusers pipeline with `compile_pipe`.
```python
import torch
from diffusers import StableDiffusionXLPipeline

from onediffx import compile_pipe

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.to("cuda")

pipe = compile_pipe(pipe)

# run once to trigger compilation
image = pipe(
    prompt="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
    height=512,
    width=512,
    num_inference_steps=30,
    output_type="pil",
).images

image[0].save(f"test_image.png")
```

### Save compiled pipeline with `save_pipe`
```python
from diffusers import StableDiffusionXLPipeline
from onediffx import compile_pipe, save_pipe
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.to("cuda")

pipe = compile_pipe(pipe)

# run once to trigger compilation
image = pipe(
    prompt="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
    height=512,
    width=512,
    num_inference_steps=30,
    output_type="pil",
).images

image[0].save(f"test_image.png")

# save the compiled pipe
save_pipe(pipe, dir="cached_pipe")
```

### Load compiled pipeline with `load_pipe`
```python
from diffusers import StableDiffusionXLPipeline
from onediffx import compile_pipe, load_pipe
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.to("cuda")

pipe = compile_pipe(pipe)

# load the compiled pipe
load_pipe(pipe, dir="cached_pipe")

# no compilation now
image = pipe(
    prompt="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
    height=512,
    width=512,
    num_inference_steps=30,
    output_type="pil",
).images

image[0].save(f"test_image.png")

```

## DeepCache speedup

### Run Stable Diffusion XL with OneDiffX

```python
import torch

from onediffx import compile_pipe
from onediffx.deep_cache import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.to("cuda")

pipe = compile_pipe(pipe)

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

### Run Stable Diffusion 1.5 with OneDiffX

```python
import torch

from onediffx import compile_pipe
from onediffx.deep_cache import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.to("cuda")

pipe = compile_pipe(pipe)

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

### Run Stable Video Diffusion with OneDiffX

```python
import torch

from diffusers.utils import load_image, export_to_video
from onediffx import compile_pipe, compile_options
from onediffx.deep_cache import StableVideoDiffusionPipeline

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.to("cuda")

compile_options.attention_allow_half_precision_score_accumulation_max_m = 0
pipe = compile_pipe(pipe, options=compile_options)

input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
input_image = input_image.resize((1024, 576))

# Warmup
for i in range(1):
    deepcache_output = pipe(
        input_image, 
        decode_chunk_size=5,
        cache_interval=3, cache_branch=0,
    ).frames[0]

deepcache_output = pipe(
    input_image, 
    decode_chunk_size=5,
    cache_interval=3, cache_branch=0,
).frames[0]

export_to_video(deepcache_output, "generated.mp4", fps=7)
```


## Fast LoRA loading and switching

OneDiff provides a more efficient implementation of loading LoRA, by invoking `load_and_fuse_lora` you can load and fuse LoRA to pipeline, and by invoking `unfuse_lora` you can restore the weight of base model.

### API

#### `onediffx.lora.load_and_fuse_lora`

`onediffx.lora.load_and_fuse_lora(pipeline: LoraLoaderMixin, pretrained_model_name_or_path_or_dict: Union[str, Path, Dict[str, torch.Tensor]], adapter_name: Optional[str] = None, *, lora_scale: float = 1.0, offload_device="cpu", offload_weight="lora", use_cache=False, **kwargs)`:
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

#### `onediffx.lora.unfuse_lora`

`onediffx.lora.unfuse_lora(pipeline: LoraLoaderMixin) -> None`:

- pipeline (`LoraLoaderMixin`): The pipeline that will unfuse LoRA weight.

#### `onediffx.lora.set_and_fuse_adapters`

`onediffx.lora.set_and_fuse_adapters(pipeline: LoraLoaderMixin, adapter_names: Union[List[str], str], adapter_weights: Optional[List[float]] = None)`

Set the LoRA layers of `adapter_name` for the unet and text-encoder(s) with related `adapter_weights`.

- pipeline (`LoraLoaderMixin`): The pipeline that will set adapters.
- adapter_names(`str` or `List[str]`): The adapter name(s) of LoRA(s) to be set for the pipeline, must appear in the `adapter_name` parameter of the `load_and_fuse_lora` function, otherwise it will be ignored.
- adapter_weights(`float` or `List[float]`, optional): The weight(s) of adapter(s), if is None, it will be set to 1.0.

#### `onediffx.lora.delete_adapters`

`onediffx.lora.delete_adapters(pipeline: LoraLoaderMixin, adapter_names: Union[List[str], str] = None)`

Deletes the LoRA layers of `adapter_name` for the unet and text-encoder(s).

- adapter_names (`str` or `List[str]`, *optional*): The names of the adapter to delete. Can be a single string or a list of strings. If is None, all adapters will be deleted.

#### `onediffx.lora.update_graph_with_constant_folding_info`

`onediffx.lora.update_graph_with_constant_folding_info(module: torch.nn.Module, info: Dict[str, flow.Tensor] = None)`

Update the weights of graph after loading LoRA. (If OneDiff has enabled constant folding optimization during compilation, some parameters in the static graph may not be updated correctly after loading lora. Invoke this function manually to update the weights of the static graph correctly.)

Check [text_to_image_sdxl_lora.py](./examples/text_to_image_sdxl_lora.py) for more details.

> **Note**: If you are using onediffx instead of diffusers and PEFT to load LoRA, there is no need to call this function, as onediffx will handle all the necessary work.

### Example

```python
import torch
from diffusers import DiffusionPipeline
from onediffx import compile_pipe
from onediffx.lora import load_and_fuse_lora, set_and_fuse_adapters, delete_adapters

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(MODEL_ID, variant="fp16", torch_dtype=torch.float16).to("cuda")
pipe = compile_pipe(pipe)

# use onediff load_and_fuse_lora
LORA_MODEL_ID = "Norod78/SDXL-YarnArtStyle-LoRA"
LORA_FILENAME = "SDXL_Yarn_Art_Style.safetensors"
load_and_fuse_lora(pipe, LORA_MODEL_ID, weight_name=LORA_FILENAME, lora_scale=1.0, adapter_name="SDXL_Yarn_Art_Style")
images_fusion = pipe(
    "a cat",
    height=1024,
    width=1024,
    generator=torch.manual_seed(0),
    num_inference_steps=30,
).images[0]
images_fusion.save("test_sdxl_lora_SDXL_Yarn_Art_Style.png")

# load another LoRA, now the pipe has two LoRA models
LORA_MODEL_ID = "ostris/watercolor_style_lora_sdxl"
LORA_FILENAME = "watercolor_v1_sdxl.safetensors"
load_and_fuse_lora(pipe, LORA_MODEL_ID, weight_name=LORA_FILENAME, lora_scale=1.0, adapter_name="watercolor")
images_fusion = pipe(
    "a cat",
    height=1024,
    width=1024,
    generator=torch.manual_seed(0),
    num_inference_steps=30,
).images[0]
images_fusion.save("test_sdxl_lora_SDXL_Yarn_Art_Style_watercolor.png")

# set LoRA 'SDXL_Yarn_Art_Style' with strength = 0.5, now the pipe has only LoRA 'SDXL_Yarn_Art_Style' with strength = 0.5
set_and_fuse_adapters(pipe, adapter_names="SDXL_Yarn_Art_Style", adapter_weights=0.5)
images_fusion = pipe(
    "a cat",
    height=1024,
    width=1024,
    generator=torch.manual_seed(0),
    num_inference_steps=30,
).images[0]
images_fusion.save("test_sdxl_lora_SDXL_Yarn_Art_Style_05.png")

# set LoRA 'SDXL_Yarn_Art_Style' with strength = 0.8 and watercolor with strength = 0.2, now the pipe has 2 LoRAs
set_and_fuse_adapters(pipe, adapter_names=["SDXL_Yarn_Art_Style", "watercolor"], adapter_weights=[0.8, 0.2])
images_fusion = pipe(
    "a cat",
    height=1024,
    width=1024,
    generator=torch.manual_seed(0),
    num_inference_steps=30,
).images[0]
images_fusion.save("test_sdxl_lora_SDXL_Yarn_Art_Style_08_watercolor_02.png")

# delete lora 'SDXL_Yarn_Art_Style', now pipe has only 'watercolor' with strength = 0.8 left
delete_adapters(pipe, "SDXL_Yarn_Art_Style")
images_fusion = pipe(
    "a cat",
    height=1024,
    width=1024,
    generator=torch.manual_seed(0),
    num_inference_steps=30,
).images[0]
images_fusion.save("test_sdxl_lora_watercolor_02.png")

```

### Benchmark

We choose 5 LoRAs to profile loading speed of 3 different APIs and switching speed of 2 different APIs, and test with and without using the PEFT backend separately. The results are shown below.

#### LoRA loading

1. `load_lora_weight`, which has high loading performance but low inference performance

2. `load_lora_weight + fuse_lora`, which has high inference performance but low loading performance

3. `onediffx.lora.load_and_fuse_lora`, which has high loading performance and high inference performance

**Without PEFT backend**

| LoRA name                               | size | HF load_lora_weight | HF load_lora_weight + fuse_lora | **OneDiffX load_and_fuse_lora** | src link                                                                |
|-----------------------------------------|------|------------------|------------------------------|---------------------------------|-------------------------------------------------------------------------|
| SDXL-Emoji-Lora-r4          | 28M  | 1.69 s           | 2.34 s                       | **0.78 s**                      | [Link](https://novita.ai/model/SDXL-Emoji-Lora-r4_160282)               |
| sdxl_metal_lora             | 23M  | 0.97 s           | 1.73 s                       | **0.19 s**                      |                                                                         |
| simple_drawing_xl_b1-000012 | 55M  | 1.67 s           | 2.57 s                       | **0.77 s**                      | [Link](https://civitai.com/models/177820/sdxl-simple-drawing)           |
| texta                       | 270M | 1.72 s           | 2.86 s                       | **0.97 s**                      | [Link](https://civitai.com/models/221240/texta-generate-text-with-sdxl) |
| watercolor_v1_sdxl_lora     | 12M  | 1.54 s           | 2.01 s                       | **0.35 s**                      |                                                                         |

**With PEFT backend**

| LoRA name                   | size   | HF load_lora_weights   | HF load_lora_weights + fuse_lora   | **OneDiffX load_and_fuse_lora**   | src link                                                                |
|:----------------------------|:-------|:--------------------|:--------------------------------|:--------------------------------|:------------------------------------------------------------------------|
| SDXL-Emoji-Lora-r4          | 28M    | 5.25 s              | 6.21 s                          | **0.78 s**                        | [Link](https://novita.ai/model/SDXL-Emoji-Lora-r4_160282)               |
| sdxl_metal_lora             | 23M    | 2.44 s              | 3.80 s                          | **0.24 s**                        |                                                                         |
| simple_drawing_xl_b1-000012 | 55M    | 4.09 s              | 5.79 s                          | **0.81 s**                        | [Link](https://civitai.com/models/177820/sdxl-simple-drawing)           |
| texta                       | 270M   | 109.13 s            | 109.71 s                        | **1.07 s**                        | [Link](https://civitai.com/models/221240/texta-generate-text-with-sdxl) |
| watercolor_v1_sdxl_lora     | 12M    | 3.08 s              | 4.04 s                          | **0.40 s**                        |                                                                         |

#### LoRA switching

We tested the performance of `set_adapters`, still using the five LoRA models mentioned above. The numbers 1-5 represent the five models 'SDXL-Emoji-Lora-r4', 'sdxl_metal_lora', 'simple_drawing_xl_b1-000012', 'texta', 'watercolor_v1_sdxl_lora'.

1. PEFT `set_adapters + fuse_lora`

2. OneDiffX `set_and_fuse_adapters`, which has the same effect as PEFT `set_adapters + fuse_lora`


| LoRA names      | PEFT set_adapters + fuse_lora   | OneDiffX set_and_fuse_adapters   |
|:----------------|:-------------------------------|:-----------------------|
| [1]             | 0.47 s                         | 0.28 s                 |
| [1, 2]          | 0.52 s                         | 0.34 s                 |
| [1, 2, 3]       | 0.71 s                         | 0.55 s                 |
| [1, 2, 3, 4]    | 2.02 s                         | 0.73 s                 |
| [1, 2, 3, 4, 5] | 1.00 s                         | 0.80 s                 |

### Note

1. OneDiff extensions for LoRA is currently only supported for limited PEFT APIs, and only supports diffusers of at least version 0.21.0.

### Optimization
- When not using the PEFT backend, diffusers will replace the module corresponding to LoRA with the LoRACompatible module, incurring additional parameter initialization time overhead. In OneDiffX, the LoRA parameters are directly fused into the model, bypassing the step of replacing the module, thereby reducing the time overhead.

- When using the PEFT backend, PEFT will also replace the module corresponding to LoRA with the corresponding BaseTunerLayer. Similar to diffusers, this increases the time overhead. OneDiffX also bypasses this step by directly operating on the original model.

- While traversing the submodules of the model, we observed that the `getattr` time overhead of OneDiff's `DeployableModule` is high. Because the parameters of DeployableModule share the same address as the PyTorch module it wraps, we choose to traverse `DeployableModule._torch_module`, greatly improving traversal efficiency.

## Compiled graph re-using

When switching models, if the new model has the same structure as the old model, you can re-use the previously compiled graph, which means you don't need to compile the new model again, which significantly reduces the time it takes you to switch models.

Here is a pseudo code, to get detailed usage, please refer to [text_to_image_sdxl_reuse_pipe](./examples/text_to_image_sdxl_reuse_pipe.py):

```python
base = StableDiffusionPipeline(...)
compiled_unet = oneflow_compile(base.unet)
base.unet = compiled_unet
# This step needs some time to compile the UNet
base(prompt)

new_base = StableDiffusionPipeline(...)
# Re-use the compiled graph by loading the new state dict into the `_torch_module` member of the object returned by `oneflow_compile`
compiled_unet._torch_module.load_state_dict(new_base.unet.state_dict())
# After loading the new state dict into the `compiled_unet._torch_module`, the weights of the compiled_unet are updated too
new_base.unet = compiled_unet
# This step doesn't need additional time to compile the UNet again because
# new_base.unet is already compiled
new_base(prompt)
```

> Note: The feature is not supported for quantized model.


## Quantization

**Note**: Quantization feature is only supported by **OneDiff Enterprise**.

OneDiff Enterprise offers a quantization method that reduces memory usage, increases speed, and maintains quality without any loss.

If you possess a OneDiff Enterprise license key, you can access instructions on OneDiff quantization and related models by visiting [Hugginface/siliconflow](https://huggingface.co/siliconflow). Alternatively, you can [contact](#contact) us to inquire about purchasing the OneDiff Enterprise license.

## Contact

For users of OneDiff Community, please visit [GitHub Issues](https://github.com/siliconflow/onediff/issues) for bug reports and feature requests.

For users of OneDiff Enterprise, you can contact contact@siliconflow.com for commercial support.

Feel free to join our [Discord](https://discord.gg/RKJTjZMcPQ) community for discussions and to receive the latest updates.
