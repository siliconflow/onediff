"""
example: python examples/text_to_image.py --height 512 --width 512 --warmup 10 --model xx
"""
import argparse
import time
import torch
import oneflow as flow
from onediff.infer_compiler import oneflow_compile
from onediff.schedulers import EulerDiscreteScheduler
from diffusers import StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Simple demo of image generation.")
    parser.add_argument(
        "--prompt", type=str, default="a photo of an astronaut riding a horse on mars"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    cmd_args = parser.parse_args()
    return cmd_args


args = parse_args()

scheduler = EulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    args.model,
    scheduler=scheduler,
    revision="fp16",
    variant="fp16",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

pipe.unet = oneflow_compile(pipe.unet)
pipe.vae = oneflow_compile(pipe.vae)

with flow.autocast("cuda"):
    for _ in range(args.warmup):
        images = pipe(
            args.prompt, height=args.height, width=args.width, num_inference_steps=args.steps
        ).images

    torch.manual_seed(args.seed)

    start_t = time.time()
    images = pipe(
        args.prompt, height=args.height, width=args.width, num_inference_steps=args.steps
    ).images
    end_t = time.time()

    cuda_memory_usage = flow._oneflow_internal.GetCUDAMemoryUsed()
    print(f"e2e ({args.steps} steps) elapsed: {end_t - start_t} s, cuda memory usage: {cuda_memory_usage} MiB")
