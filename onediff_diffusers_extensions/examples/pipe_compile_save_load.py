# Command to run oneflow backend save: python pipe_compile_save_load.py --save
# Command to run oneflow backend load: python pipe_compile_save_load.py --load
# Command to run nexfort backend save/load: python pipe_compile_save_load.py --compiler nexfort
import json
import time
import argparse

import torch
from diffusers import StableDiffusionXLPipeline
from onediffx import (
    compile_pipe,
    save_pipe,
    load_pipe,
    setup_nexfort_pipe_cache,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
)
parser.add_argument("--save", action=argparse.BooleanOptionalAction)
parser.add_argument("--load", action=argparse.BooleanOptionalAction)
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
parser.add_argument("--warmup-iterations", type=int, default=1)
args = parser.parse_args()

pipe = StableDiffusionXLPipeline.from_pretrained(
    args.model, torch_dtype=torch.float16, use_safetensors=True
)
pipe.to("cuda")

# Compile the pipe
if args.compiler == "oneflow":
    pipe = compile_pipe(pipe)
else:
    if args.compiler_config is not None:
        options = json.loads(args.compiler_config)
    else:
        options = '{"mode": "max-optimize:max-autotune:freezing:benchmark:cudagraphs", "memory_format": "channels_last"}'

    setup_nexfort_pipe_cache("nexfort_cached_pipe")
    pipe = compile_pipe(
        pipe, backend="nexfort", options=options, fuse_qkv_projections=True
    )

if args.load:
    # Load the compiled pipe
    load_pipe(pipe, dir="oneflow_cached_pipe")

# Warm-up iterations
start_time = time.time()
for _ in range(args.warmup_iterations):
    pipe(
        prompt="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
        height=512,
        width=512,
        num_inference_steps=30,
        output_type="pil",
    )
warmup_time = time.time() - start_time
print(f"Warmup Time: {warmup_time:.2f} seconds")

# If the pipe is not loaded, it will takes seconds to do real compilation.
# If the pipe is loaded, it will run fast.
image = pipe(
    prompt="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
    height=512,
    width=512,
    num_inference_steps=30,
    output_type="pil",
).images

image[0].save(f"test_image.png")

if args.save:
    # Save the compiled pipe
    save_pipe(pipe, dir="oneflow_cached_pipe")
