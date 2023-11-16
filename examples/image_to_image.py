from PIL import Image


import argparse
from onediff.infer_compiler import oneflow_compile
from diffusers import StableDiffusionImg2ImgPipeline
import oneflow as flow
import torch


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

PROMPT = "sea,beach,the waves crashed on the sand,blue sky whit white cloud"

img = Image.new("RGB", (512, 512), "#1f80f0")

with flow.autocast("cuda"):
    images = pipe(
        PROMPT,
        image=img,
        guidance_scale=10,
        num_inference_steps=100,
        output_type="np",
    ).images
    for i, image in enumerate(images):
        pipe.numpy_to_pil(image)[0].save(f"{PROMPT}-of-{i}.png")
