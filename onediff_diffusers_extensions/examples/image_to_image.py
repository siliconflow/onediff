import argparse

import torch
from PIL import Image
import oneflow as flow  # usort: skip

from diffusers import StableDiffusionImg2ImgPipeline
from onediff.infer_compiler import oneflow_compile


prompt = "sea,beach,the waves crashed on the sand,blue sky whit white cloud"


def parse_args():
    parser = argparse.ArgumentParser(description="Simple demo of image generation.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
    )
    cmd_args = parser.parse_args()
    return cmd_args


args = parse_args()


pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    args.model_id,
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
)

pipe = pipe.to("cuda")
pipe.unet = oneflow_compile(pipe.unet)
pipe.vae.decoder = oneflow_compile(pipe.vae.decoder)


img = Image.new("RGB", (512, 512), "#1f80f0")

with flow.autocast("cuda"):
    images = pipe(
        prompt,
        image=img,
        guidance_scale=10,
        num_inference_steps=100,
        output_type="np",
    ).images
    for i, image in enumerate(images):
        pipe.numpy_to_pil(image)[0].save(f"{prompt}-of-{i}.png")
