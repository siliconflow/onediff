"""
Torch run example: python examples/text_to_image_sdxl.py
Compile to oneflow graph example: python examples/text_to_image_sdxl.py --compile
"""
import os
import argparse

import torch
import oneflow as flow


from onediff.infer_compiler import oneflow_compile
from onediff.schedulers import EulerDiscreteScheduler
from onediff.optimization import rewrite_self_attention
from diffusers import StableDiffusionXLPipeline

from onediff.infer_compiler.utils.cost_util import cost_cnt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base", type=str, default="/share_nfs/hf_models/stable-diffusion-xl-base-1.0/"
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
    "--compile", type=(lambda x: str(x).lower() in ["true", "1", "yes"]), default=True
)
### add save and load graph ###
parser.add_argument("--save", action=argparse.BooleanOptionalAction)
parser.add_argument("--load", action=argparse.BooleanOptionalAction)
parser.add_argument("--file", type=str, required=False, default="unet_compiled")
### end ###

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
if args.compile:
    print("unet is compiled to oneflow.")
    rewrite_self_attention(base.unet)
    base.unet = oneflow_compile(base.unet)
    if args.load:
        @cost_cnt
        def _load_graph():
            print(f"Loading graph from {args.file}")
            base.unet.load_graph(args.file)
        _load_graph()

# Warmup
for i in range(args.warmup):
    image = base(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
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

# Save graph
if args.save:
    @cost_cnt
    def _save_graph():
        print(f"Saving graph to {args.file}")
        base.unet.save_graph(args.file)
    _save_graph()
