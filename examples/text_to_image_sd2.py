import os

import cv2
import torch
from onediff.infer_compiler import oneflow_compile
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

model_id = "/share_nfs/hf_models/stable-diffusion-2-1"
# you can also use a local file directory like this here
# model_id = "/share_nfs/hf_models/stable-diffusion-2-1"
# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
)

if "upcast_attention" in pipe.unet.config and pipe.unet.config.upcast_attention:
    os.environ["ONEFLOW_KERENL_FMHA_ENABLE_TRT_FLASH_ATTN_IMPL"] = "0"
else:
    os.environ["ONEFLOW_KERENL_FMHA_ENABLE_TRT_FLASH_ATTN_IMPL"] = "1"

pipe = pipe.to("cuda")
pipe.unet = oneflow_compile(pipe.unet)
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, height=512, width=512).images[0]

image.save("astronaut_rides_horse.png")
