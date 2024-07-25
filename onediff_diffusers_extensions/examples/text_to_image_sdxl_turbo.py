"""
Torch run example: python examples/text_to_image_sdxl_turbo.py
Compile to oneflow graph example: python examples/text_to_image_sdxl_turbo.py --compile
"""
import argparse
import os
import time

import torch
import oneflow as flow  # usort: skip

from diffusers import AutoPipelineForText2Image
from onediff.infer_compiler import oneflow_compile

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str, default="stabilityai/sdxl-turbo")
parser.add_argument("--variant", type=str, default="fp16")
parser.add_argument(
    "--prompt",
    type=str,
    default="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--n_steps", type=int, default=4)
parser.add_argument("--saved_image", type=str, required=False, default="sdxl-out.png")
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--compile", type=(lambda x: str(x).lower() in ["true", "1", "yes"]), default=True
)
args = parser.parse_args()

# Normal SDXL turbo pipeline init.
OUTPUT_TYPE = "pil"

# SDXL turbo base: AutoPipelineForText2Image
base = AutoPipelineForText2Image.from_pretrained(
    args.base,
    torch_dtype=torch.float16,
    variant=args.variant,
    use_safetensors=True,
)
base.to("cuda")

# Compile unet with oneflow
if args.compile:
    print("unet is compiled to oneflow.")
    base.unet = oneflow_compile(base.unet)
    print("vae is compiled to oneflow.")
    base.vae.decoder = oneflow_compile(base.vae.decoder)

# Warmup
for i in range(args.warmup):
    torch.manual_seed(args.seed)
    image = base(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.n_steps,
        guidance_scale=0.0,
        output_type=OUTPUT_TYPE,
    ).images

# Normal SDXL turbo run
torch.manual_seed(args.seed)
start_t = time.time()

image = base(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.n_steps,
    guidance_scale=0.0,
    output_type=OUTPUT_TYPE,
).images
end_t = time.time()
print(f"e2e ({args.n_steps} steps) elapsed: {end_t - start_t} s")

image[0].save(f"h{args.height}-w{args.width}-{args.saved_image}")
