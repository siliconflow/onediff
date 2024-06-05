"""
Torch run example: python examples/text_to_image_sdxl.py
Compile to oneflow graph example: python examples/text_to_image_sdxl.py
"""
import os
import argparse

import torch
import oneflow as flow

from onediff.infer_compiler import oneflow_compile
from onediff.schedulers import EulerDiscreteScheduler
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
    "--run_multiple_resolutions",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=True,
)
parser.add_argument(
    "--run_rare_resolutions",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=True,
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
    base.unet = oneflow_compile(base.unet)

# Compile vae with oneflow
if args.compile_vae:
    print("Compiling vae with oneflow.")
    base.vae.decoder = oneflow_compile(base.vae.decoder)

# Warmup with run
# Will do compilatioin in the first run
print("Warmup with running graphs...")
torch.manual_seed(args.seed)
image = base(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.n_steps,
    output_type=OUTPUT_TYPE,
).images

# Normal SDXL run
print("Normal SDXL run...")
torch.manual_seed(args.seed)
image = base(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.n_steps,
    output_type=OUTPUT_TYPE,
).images
image[0].save(f"h{args.height}-w{args.width}-{args.saved_image}")


# Should have no compilation for these new input shape
if args.run_multiple_resolutions:
    print("Test run with multiple resolutions...")
    sizes = [960, 720, 896, 768]
    if "CI" in os.environ:
        sizes = [360]
    for h in sizes:
        for w in sizes:
            image = base(
                prompt=args.prompt,
                height=h,
                width=w,
                num_inference_steps=args.n_steps,
                output_type=OUTPUT_TYPE,
            ).images


if args.run_rare_resolutions:
    print("Test run with other another uncommon resolution...")
    h = 544
    w = 408
    image = base(
        prompt=args.prompt,
        height=h,
        width=w,
        num_inference_steps=args.n_steps,
        output_type=OUTPUT_TYPE,
    ).images
