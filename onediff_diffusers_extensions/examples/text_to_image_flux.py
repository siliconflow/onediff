import argparse
import time

import cv2
import numpy as np
import torch

from diffusers import FluxPipeline
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str, default="black-forest-labs/FLUX.1-schnell")
parser.add_argument(
    "--prompt",
    type=str,
    default="chinese painting style women",
)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--n_steps", type=int, default=4)
parser.add_argument("--saved_image", type=str, required=False, default="flux-out.png")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--run", type=int, default=3)
parser.add_argument(
    "--compile", type=(lambda x: str(x).lower() in ["true", "1", "yes"]), default=True
)
parser.add_argument("--run-multiple-resolutions", action="store_true")
args = parser.parse_args()


# load stable diffusion
pipe = FluxPipeline.from_pretrained(args.base, torch_dtype=torch.bfloat16)
# pipe = FluxPipeline.from_pretrained(args.base, torch_dtype=torch.bfloat16, local_files_only=True, revision="93424e3a1530639fefdf08d2a7a954312e5cb254")
pipe.to("cuda")

if args.compile:
    from onediffx import compile_pipe

    pipe = compile_pipe(
        pipe,
        backend="nexfort",
        options={
            "options": {
                "cuda.fuse_timestep_embedding": False,
                "inductor.force_triton_sdpa": True,
            }
        },
    )


# generate image
generator = torch.manual_seed(args.seed)

print("Warmup")
for i in range(args.warmup):
    image = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        output_type="pil",
        num_inference_steps=args.n_steps,  # use a larger number if you are using [dev]
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]


print("Run")
for i in range(args.run):
    begin = time.time()
    image = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        output_type="pil",
        num_inference_steps=args.n_steps,  # use a larger number if you are using [dev]
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    end = time.time()
    print(f"Inference time: {end - begin:.3f}s")

    image.save(f"{i=}th_{args.saved_image}.png")


if args.run_multiple_resolutions:
    print("Test run with multiple resolutions...")
    sizes = [1024, 512, 768, 256]
    for h in sizes:
        for w in sizes:
            print(f"Running at resolution: {h}x{w}")
            start_time = time.time()
            image = pipe(
                args.prompt,
                height=h,
                width=w,
                output_type="pil",
                num_inference_steps=args.n_steps,  # use a larger number if you are using [dev]
                generator=torch.Generator("cpu").manual_seed(args.seed),
            ).images[0]
            end_time = time.time()
            print(f"Inference time: {end_time - start_time:.2f} seconds")
    image.save(f"{i=}th_{args.saved_image}_{h}x{w}.png")
