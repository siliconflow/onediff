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
        "--output_dir",
        type=str,
        default="oneflow-sd-output",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
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

os.makedirs(args.output_dir, exist_ok=True)
prompt = "a photo of an astronaut riding a horse on mars"
for n in range(args.n):
    images = pipe(args.prompt).images
    for i, image in enumerate(images):
        dst = os.path.join(args.output_dir, f"{prompt[:100]}-{n}-{i}.png")
        image.save(dst)
