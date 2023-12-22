# A acceleration library for DeepCache algorithm

## Reference

DeepCache originality is here https://github.com/horseee/DeepCache
```
@article{ma2023deepcache,
  title={DeepCache: Accelerating Diffusion Models for Free},
  author={Ma, Xinyin and Fang, Gongfan and Wang, Xinchao},
  journal={arXiv preprint arXiv:2312.00858},
  year={2023}
}
```

## Quick Start

1. Install differs_extensions

    - Follow the steps [here](https://github.com/Oneflow-Inc/onediff?tab=readme-ov-file#install-from-source) to install onediff. 

    - Install differs_extensions by following these steps
        ```
        git clone https://github.com/Oneflow-Inc/onediff.git
        cd differs_extensions && python3 -m pip install -e .
        ```

2. Usage
    - Run Stable Diffusion XL with the this acceleration library

        ```
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

        base.unet = oneflow_compile(base.unet)
        base.fast_unet = oneflow_compile(base.fast_unet)
        base.vae = oneflow_compile(base.vae)

        prompt = "A photo of a cat. Focus light and create sharp, defined edges."        
        deepcache_output = pipe(
            prompt, 
            cache_interval=3, cache_layer_id=0, cache_block_id=0,
            output_type='pil'
        ).images[0]

        ```

    - Run Stable Diffusion with the this acceleration library

        ```
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

        base.unet = oneflow_compile(base.unet)
        base.fast_unet = oneflow_compile(base.fast_unet)
        base.vae = oneflow_compile(base.vae)

        prompt = "a photo of an astronaut on a moon"       
        deepcache_output = pipe(
            prompt, 
            cache_interval=3, cache_layer_id=0, cache_block_id=0,
            output_type='pil'
        ).images[0]

        ```
