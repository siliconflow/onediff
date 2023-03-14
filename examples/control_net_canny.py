import sys
import os
import cv2
from PIL import Image
import numpy as np

sys.path.append("../diffusers/src")
from onediff import OneFlowStableDiffusionControlNetPipeline

from diffusers.utils import load_image
from diffusers import ControlNetModel

import oneflow as flow
flow.mock_torch.enable()

image = load_image(
    "http://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)


image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)


controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=flow.float16)
    
pipe = OneFlowStableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=flow.float16
)

# this command loads the individual model components on GPU on-demand.
pipe.enable_model_cpu_offload()

generator = flow.manual_seed(0)

prompt = "disco dancer with colorful lights, best quality, extremely detailed"
negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"


out_images = pipe(
    prompt = prompt, negative_prompt=negative_prompt, num_inference_steps=20, generator=generator, image=canny_image
).images
for i, image in enumerate(out_images):
    image.save(f"{prompt}-of-{i}.png")
