# A acceleration library for diffusers

## Install diffusers_extensions

- Follow the steps [here](https://github.com/Oneflow-Inc/onediff?tab=readme-ov-file#install-from-source) to install onediff. 

- Install diffusers_extensions by following these steps
    ```
    git clone https://github.com/Oneflow-Inc/onediff.git
    cd diffusers_extensions && python3 -m pip install -e .
    ```

## DeepCache Speedup

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
