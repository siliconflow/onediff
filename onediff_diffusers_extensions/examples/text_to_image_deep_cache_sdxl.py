"""
Torch run example: python examples/text_to_image_deep_cache_sdxl.py --compile 0
Compile to oneflow graph example: python examples/text_to_image_deep_cache_sdxl.py
"""
import argparse
import os

import torch

from onediffx import compile_pipe
from onediffx.deep_cache import StableDiffusionXLPipeline

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
    "--compile",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=True,
)
parser.add_argument(
    "--run_multiple_resolutions",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=False,
)
args = parser.parse_args()

# Normal SDXL pipeline init.
OUTPUT_TYPE = "pil"

# SDXL base: StableDiffusionXLPipeline
base = StableDiffusionXLPipeline.from_pretrained(
    args.base,
    torch_dtype=torch.float16,
    variant=args.variant,
    use_safetensors=True,
)
base.to("cuda")

# Compile unet with oneflow
if args.compile:
    print("Compiling unet with oneflow.")
    base = compile_pipe(base)


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
    cache_interval=3,
    cache_layer_id=0,
    cache_block_id=0,
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
    cache_interval=3,
    cache_layer_id=0,
    cache_block_id=0,
).images
image[0].save(f"h{args.height}-w{args.width}-{args.saved_image}")

print("Test run with multiple resolutions...")
if args.run_multiple_resolutions:
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
                cache_interval=3,
                cache_layer_id=0,
                cache_block_id=0,
            ).images
