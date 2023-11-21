"""
example: python examples/text_to_image.py --height 512 --width 512 --warmup 10 --model_id xx
"""
import argparse
from onediff.infer_compiler import oneflow_compile, oneflow_load_compiled
from onediff.schedulers import EulerDiscreteScheduler
from onediff.optimization import rewrite_self_attention
from diffusers import StableDiffusionPipeline
import oneflow as flow
import torch
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Simple demo of image generation.")
    parser.add_argument("--compiled_graph_path", type=str, default="compiled-graph")
    parser.add_argument(
        "--prompt", type=str, default="an icon of a cat"
    )
    parser.add_argument(
        "--model_id",
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

scheduler = EulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    args.model_id,
    scheduler=scheduler,
    use_auth_token=True,
    revision="fp16",
    variant="fp16",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe = pipe.to("cuda")

compiled_graph_exists = os.path.exists(args.compiled_graph_path)

if not compiled_graph_exists:
    rewrite_self_attention(pipe.unet)
    pipe.unet = oneflow_compile(pipe.unet)
else:
    pipe.unet = oneflow_load_compiled(pipe.unet, args.compiled_graph_path, device="cuda")

prompt = args.prompt
with flow.autocast("cuda"):
    torch.manual_seed(args.seed)

    images = pipe(
        prompt, height=args.height, width=args.width, num_inference_steps=args.steps
    ).images

    if not compiled_graph_exists:
        print("Saving compiled graph")
        pipe.unet.save_graph(args.compiled_graph_path)

    for i, image in enumerate(images):
        image.save(f"{prompt}-of-{i}.png")
