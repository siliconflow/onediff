""" 
performs image generation using a stable diffusion model with a control network. 
"""
import cv2
from onediff.infer_compiler import oneflow_compile
from PIL import Image
import numpy as np


import oneflow as flow
from diffusers.utils import load_image
from diffusers import ControlNetModel
from diffusers import StableDiffusionControlNetPipeline
import torch


image = load_image(
    "http://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)

image = np.array(image)

LOW_THRESHOLD = 100
HIGH_THRESHOLD = 200
PROMPT = "disco dancer with colorful lights, best quality, extremely detailed"
NEGATIVE_PROMPT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

image = cv2.Canny(image, LOW_THRESHOLD, HIGH_THRESHOLD)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.to("cuda")
pipe.unet = oneflow_compile(pipe.unet)
generator = torch.manual_seed(0)


out_images = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    num_inference_steps=20,
    generator=generator,
    image=canny_image,
).images
for i, image in enumerate(out_images):
    image.save(f"{PROMPT}-of-{i}.png")
