"""
Compile to oneflow graph with :
oneflow_compile example: python examples/text_to_image_sdxl.py --compile
torch.compile example: python examples/text_to_image_sdxl.py --compile_with_dynamo
"""
import os
import argparse

import oneflow as flow
import torch

from diffusers import DiffusionPipeline
from onediff.infer_compiler import oneflow_compile

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
)
parser.add_argument(
    "--refiner", type=str, default="stabilityai/stable-diffusion-xl-refiner-1.0"
)
parser.add_argument("--variant", type=str, default="fp16")
parser.add_argument(
    "--prompt",
    type=str,
    default="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
)
parser.add_argument("--n_steps", type=int, default=1)
parser.add_argument("--saved_image", type=str, required=False, default="sdxl-out.png")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--compile", type=(lambda x: str(x).lower() in ["true", "1", "yes"]), default=True
)
parser.add_argument("--compile_with_dynamo", action=argparse.BooleanOptionalAction)
parser.add_argument("--num_dynamic_input_size", type=int, default=9)
cmd_args = parser.parse_args()

if cmd_args.compile and cmd_args.compile_with_dynamo:
    parser.error("--compile and --compile_with_dynamo cannot be used together.")

# Normal SDXL pipeline init.
SEED = torch.Generator("cuda").manual_seed(cmd_args.seed)
OUTPUT_TYPE = "pil"
# SDXL base: StableDiffusionXLPipeline
base = DiffusionPipeline.from_pretrained(
    cmd_args.base,
    torch_dtype=torch.float16,
    variant=cmd_args.variant,
    use_safetensors=True,
)
base.to("cuda")

# Compile unet with oneflow
if cmd_args.compile:
    print("unet is compiled to oneflow.")
    base.unet = oneflow_compile(
        base.unet, options={"size": cmd_args.num_dynamic_input_size}
    )

# Compile unet with torch.compile to oneflow.
# Note this is at alpha stage(experimental) and may be changed later.
if cmd_args.compile_with_dynamo:
    print("unet is compiled to oneflow with torch.compile.")
    from onediff.infer_compiler import oneflow_backend

    base.unet = torch.compile(
        base.unet, fullgraph=True, mode="reduce-overhead", backend=oneflow_backend
    )

# Normal SDXL run
# sizes = [1024, 896, 768]
sizes = [1024]
for h in sizes:
    for w in sizes:
        for i in range(3):
            image = base(
                prompt=cmd_args.prompt,
                height=h,
                width=w,
                generator=SEED,
                num_inference_steps=cmd_args.n_steps,
                output_type=OUTPUT_TYPE,
            ).images
            image[0].save(f"h{h}-w{w}-i{i}-{cmd_args.saved_image}")
