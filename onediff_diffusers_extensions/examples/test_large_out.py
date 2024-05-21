def test_fn():
    import torch
    from diffusers import StableDiffusionXLPipeline
    from onediff.infer_compiler import oneflow_compile
    t2i_pipe = StableDiffusionXLPipeline.from_pretrained(
        "/data/hf_models/sdxl/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    t2i_pipe.to("cuda")
    #t2i_pipe.vae.decoder = oneflow_compile(t2i_pipe.vae.decoder)
    t2i_pipe.unet = oneflow_compile(t2i_pipe.unet)
    t2i_pipe(
        prompt="a photo of a cat",
        negative_prompt="",
        num_inference_steps=30,
        height=2048,
        width=2048,
    )
test_fn()