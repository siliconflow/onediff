"""
example: python3 examples/text_to_video.py
"""
import time
import argparse
from PIL import Image

import torch
import oneflow as flow

from onediff.infer_compiler import oneflow_compile
from onediff.infer_compiler.utils import set_boolean_env_var
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video


set_boolean_env_var("ONEFLOW_KERENL_FMHA_ENABLE_TRT_FLASH_ATTN_IMPL", False)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video using Stable Video Diffusion."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="examples/assets/rocket.png",  # Replace with your local image path
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--decode_chunk_size", type=int, default=8)
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--output_file", type=str, default="generated.mp4")
    return parser.parse_args()

args = parse_args()

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = pipe.to("cuda")
pipe.unet = oneflow_compile(pipe.unet)

# Load the conditioning image from local file
image = Image.open(args.image_path)
image = image.resize((1024, 576))

with flow.autocast("cuda"):
    generator = torch.manual_seed(args.seed)

    start_t = time.time()
    # Warm-up
    for _ in range(args.warmup):
        _ = pipe(image, decode_chunk_size=args.decode_chunk_size, generator=generator, num_frames=25)
    end_t = time.time()
    print(f"warm-up elapsed: {end_t - start_t} s")

    start_t = time.time()
    frames = pipe(
        image, decode_chunk_size=args.decode_chunk_size, generator=generator, num_frames=25
    ).frames[0]

export_to_video(frames, args.output_file, fps=args.fps)
end_t = time.time()
print(f"e2e elapsed: {end_t - start_t} s")
