# Command to run save: python test_pipe_compile_save_load.py --save
# Command to run load: python test_pipe_compile_save_load.py --load
import argparse

import torch
from diffusers import StableDiffusionXLPipeline
from onediffx import compile_pipe, load_pipe, save_pipe

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
)
parser.add_argument("--save", action=argparse.BooleanOptionalAction)
parser.add_argument("--load", action=argparse.BooleanOptionalAction)
cmd_args = parser.parse_args()

pipe = StableDiffusionXLPipeline.from_pretrained(
    "/share_nfs/hf_models/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.to("cuda")

# compile the pipe
pipe = compile_pipe(pipe)

if cmd_args.load:
    # load the compiled pipe
    load_pipe(pipe, dir="cached_pipe")

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

if cmd_args.save:
    # save the compiled pipe
    save_pipe(pipe, dir="cached_pipe")
