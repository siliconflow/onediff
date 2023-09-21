from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from onediff.infer_compiler import oneflow_compile

# you can also use a local file directory like this here
model_id = "/share_nfs/hf_models/stable-diffusion-2-1"
model_id = "stabilityai/stable-diffusion-2"
# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
pipe.unet = oneflow_compile(pipe.unet)
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, height=768, width=768).images[0]

image.save("astronaut_rides_horse.png")
