"""
Torch run example: python examples/text_to_image_sdxl.py
Compile with oneflow: python examples/text_to_image_sdxl.py --compiler oneflow
Compile with nexfort: python examples/text_to_image_sdxl.py --compiler nexfort
Test dynamic shape: Add --run_multiple_resolutions 1 and --run_rare_resolutions 1
"""

import argparse
import json
import os
import time

import torch
import oneflow as flow  # usort: skip

from diffusers import StableDiffusionXLPipeline

from onediff.infer_compiler import oneflow_compile
from onediff.schedulers import EulerDiscreteScheduler
from onediffx import compile_pipe

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
# parser.add_argument(
#     "--compile_unet",
#     type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
#     default=True,
# )
# parser.add_argument(
#     "--compile_vae",
#     type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
#     default=True,
# )
parser.add_argument(
    "--compiler",
    type=str,
    default="oneflow",
    choices=["oneflow", "nexfort"],
)
parser.add_argument(
    "--compiler-config",
    type=str,
    default=None,
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

# # Compile unet with oneflow
# if args.compile_unet:
#     print("Compiling unet with oneflow.")
#     base.unet = oneflow_compile(base.unet)

# # Compile vae with oneflow
# if args.compile_vae:
#     print("Compiling vae with oneflow.")
#     base.vae.decoder = oneflow_compile(base.vae.decoder)

# Compile the pipe
if args.compiler == "oneflow":
    base.unet = oneflow_compile(base.unet)
elif args.compiler == "nexfort":
    if args.compiler_config is not None:
        options = json.loads(args.compiler_config)
    else:
        options = json.loads('{"mode": "max-autotune:cudagraphs", "dynamic": true}')

    base = compile_pipe(
        base, backend="nexfort", options=options, fuse_qkv_projections=True
    )

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
print(f"Running at resolution: {args.height}x{args.width}")
start_time = time.time()
image = base(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.n_steps,
    output_type=OUTPUT_TYPE,
).images
end_time = time.time()
print(f"Inference time: {end_time - start_time:.2f} seconds")
image[0].save(f"h{args.height}-w{args.width}-{args.saved_image}")


# Should have no compilation for these new input shape
# The nexfort backend encounters an exception when dynamically switching resolution to 960x720.
if args.run_multiple_resolutions:
    print("Test run with multiple resolutions...")
    sizes = [960, 720, 896, 768]
    if "CI" in os.environ:
        sizes = [360]
    for h in sizes:
        for w in sizes:
            print(f"Running at resolution: {h}x{w}")
            start_time = time.time()
            image = base(
                prompt=args.prompt,
                height=h,
                width=w,
                num_inference_steps=args.n_steps,
                output_type=OUTPUT_TYPE,
            ).images
            end_time = time.time()
            print(f"Inference time: {end_time - start_time:.2f} seconds")


if args.run_rare_resolutions:
    print("Test run with other another uncommon resolution...")
    h = 544
    w = 408
    print(f"Running at resolution: {h}x{w}")
    start_time = time.time()
    image = base(
        prompt=args.prompt,
        height=h,
        width=w,
        num_inference_steps=args.n_steps,
        output_type=OUTPUT_TYPE,
    ).images
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds")
