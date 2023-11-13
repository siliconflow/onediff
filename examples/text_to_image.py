import argparse
from onediff.infer_compiler import oneflow_compile
from onediff.schedulers import EulerDiscreteScheduler
from onediff.optimization import rewrite_self_attention
from diffusers import StableDiffusionPipeline
import oneflow as flow
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Simple demo of image generation.")
    parser.add_argument(
        "--prompt", type=str, default="a photo of an astronaut riding a horse on mars"
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
    args = parser.parse_args()
    return args


args = parse_args()

pipe = StableDiffusionPipeline.from_pretrained(
    args.model_id,
    use_auth_token=True,
    revision="fp16",
    variant="fp16",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe = pipe.to("cuda")

rewrite_self_attention(pipe.unet)
pipe.unet = oneflow_compile(pipe.unet)

prompt = args.prompt
with flow.autocast("cuda"):
    for _ in range(args.warmup):
        images = pipe(
            prompt, height=args.height, width=args.width, num_inference_steps=args.steps
        ).images

    torch.manual_seed(args.seed)

    images = pipe(
        prompt, height=args.height, width=args.width, num_inference_steps=args.steps
    ).images
    for i, image in enumerate(images):
        image.save(f"{prompt}-of-{i}.png")
