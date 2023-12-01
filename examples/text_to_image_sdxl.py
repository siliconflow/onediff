"""
Torch run example: python examples/text_to_image_sdxl.py
Compile to oneflow graph example: python examples/text_to_image_sdxl.py
"""
import os
import argparse

import oneflow as flow
import torch

from onediff.infer_compiler import oneflow_compile
from onediff.schedulers import EulerDiscreteScheduler
from onediff.optimization import rewrite_self_attention
from diffusers import StableDiffusionXLPipeline

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
)
parser.add_argument("--variant", type=str, default="fp16")
parser.add_argument(
    "--prompt",
    type=str,
    default="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--n_steps", type=int, default=30)
parser.add_argument("--saved_image", type=str, required=False, default="sdxl-out.png")
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--compile_unet",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=True,
)
parser.add_argument(
    "--compile_vae",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=True,
)
parser.add_argument(
    "--use_multiple_resolutions",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=False,
)
args = parser.parse_args()

# Normal SDXL pipeline init.
OUTPUT_TYPE = "pil"

# SDXL base: StableDiffusionXLPipeline
scheduler = EulerDiscreteScheduler.from_pretrained(args.base, subfolder="scheduler")
base = StableDiffusionXLPipeline.from_pretrained(
    args.base,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    variant=args.variant,
    use_safetensors=True,
)
base.to("cuda")

# Compile unet with oneflow
if args.compile_unet:
    print("Compiling unet with oneflow.")
    rewrite_self_attention(base.unet)
    base.unet = oneflow_compile(base.unet)

# Compile vae with oneflow
if args.compile_vae:
    print("Compiling vae with oneflow.")
    base.vae = oneflow_compile(base.vae)

# Define multiple resolutions for warmup
resolutions = (
    [(512, 512), (256, 256), ]
    if args.use_multiple_resolutions
    else [(args.height, args.width)]
)

# Warmup with chosen resolutions
for resolution in resolutions:
    for i in range(args.warmup):
        image = base(
            prompt=args.prompt,
            height=resolution[0],
            width=resolution[1],
            num_inference_steps=args.n_steps,
            output_type=OUTPUT_TYPE,
        ).images

# Normal SDXL run
torch.manual_seed(args.seed)
image = base(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.n_steps,
    output_type=OUTPUT_TYPE,
).images
image[0].save(f"h{args.height}-w{args.width}-{args.saved_image}")
