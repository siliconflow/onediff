# HF_HUB_OFFLINE=1 python3 examples/torch_interpretor.py
import os
import torch
from diffusers import StableDiffusionPipeline
from onediff.infer_compiler import oneflow_backend

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
)

# run with interpreter mode to oneflow
# ONFLOW_INTERPRETER's default value is 0
os.environ["ONFLOW_INTERPRETER"] = "0"

# optimize with oneflow graph
# ONEFLOW_GRAPH's default value is 0
os.environ["ONEFLOW_GRAPH"] = "1"


pipe.unet = torch.compile(
    pipe.unet, fullgraph=True, mode="reduce-overhead", backend=oneflow_backend
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with torch.autocast("cuda"):
    for i in range(3):
        images = pipe(prompt).images
        for j, image in enumerate(images):
            image.save(f"{prompt}-of-{i}-{j}.png")
