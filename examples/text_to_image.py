import argparse
from onediff.infer_compiler import oneflow_compile
from diffusers import StableDiffusionPipeline
import oneflow as flow
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Simple demo of image generation.")
    parser.add_argument(
        "--prompt", type=str, default="a photo of an astronaut riding a horse on mars"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    args = parser.parse_args()
    return args


args = parse_args()


pipe = StableDiffusionPipeline.from_pretrained(
    args.model_id,
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
)

pipe = pipe.to("cuda")
pipe.unet = oneflow_compile(pipe.unet)

prompt = args.prompt
with flow.autocast("cuda"):
    images = pipe(prompt).images
    for i, image in enumerate(images):
        image.save(f"{prompt}-of-{i}.png")
