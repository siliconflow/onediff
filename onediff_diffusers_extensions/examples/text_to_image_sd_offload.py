"""
# run sd 1.5
ONEDIFF_DEBUG=1 python text_to_image_sd_offload.py --base /share_nfs/hf_models/stable-diffusion-v1-5

# run sdxl
ONEDIFF_DEBUG=1 python text_to_image_sd_offload.py --base /share_nfs/hf_models/stable-diffusion-xl-base-1.0
"""
import os
import argparse

import oneflow as flow
import torch

from onediff.infer_compiler import oneflow_compile
from onediff.schedulers import EulerDiscreteScheduler
from diffusers import AutoPipelineForText2Image

from onediff.infer_compiler.utils.cost_util import cost_cnt

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
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--n_steps", type=int, default=10)
parser.add_argument("--saved_image", type=str, required=False, default="sd-out.png")
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
args = parser.parse_args()

# Normal SD pipeline init.
OUTPUT_TYPE = "pil"

scheduler = EulerDiscreteScheduler.from_pretrained(args.base, subfolder="scheduler")
base = AutoPipelineForText2Image.from_pretrained(
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
# if args.compile_vae:
#     print("Compiling vae with oneflow.")
#     base.vae.decoder = oneflow_compile(base.vae.decoder)

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

# Normal SD run
@cost_cnt(True)
def run():
    print("Normal SD run...")
    torch.manual_seed(args.seed)
    image = base(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.n_steps,
        output_type=OUTPUT_TYPE,
    ).images
    image[0].save(f"h{args.height}-w{args.width}-{args.saved_image}")

@cost_cnt(True)
def offload():
    print("offload graph to CPU")
    base.unet.offload()

@cost_cnt(True)
def load():
    print("load graph to GPU")
    base.unet.load()

run()
offload()
load()
run()
