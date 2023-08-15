# HF_HUB_OFFLINE=1 python3 examples/torch_interpretor.py
import torch
from diffusers import StableDiffusionPipeline
from onediff.infer_compiler import torchbackend

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
)

pipe.unet = torch.compile(pipe.unet, fullgraph=True, mode="reduce-overhead", backend=torchbackend)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with torch.autocast("cuda"):
    images = pipe(prompt).images
    for i, image in enumerate(images):
        image.save(f"{prompt}-of-{i}.png")
