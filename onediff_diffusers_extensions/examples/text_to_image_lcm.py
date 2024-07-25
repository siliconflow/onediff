import argparse
import importlib.metadata

from diffusers import DiffusionPipeline
from packaging import version


def check_diffusers_version():
    required_version = version.parse("0.22.0")
    package_name = "diffusers"

    try:
        installed_version = version.parse(importlib.metadata.version(package_name))
        if installed_version < required_version:
            raise ValueError(
                f"Installed {package_name} version ({installed_version}) is lower than required ({required_version})"
            )

    except importlib.metadata.PackageNotFoundError:
        print(f"{package_name} is not installed")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple demo of LCM image generation.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="SimianLuo/LCM_Dreamshaper_v7",
        help="Model id or local path to the LCM model.",
    )
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup option sets the number of preliminary runs of the pipeline. These initial runs are necessary to ensure accurate performance data during testing.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--disable", action="store_true", help="Disable Onediff speeding up."
    )
    args = parser.parse_args()
    return args


check_diffusers_version()

args = parse_args()

if not args.disable:
    from onediff.infer_compiler import oneflow_compile
import torch

pipe = DiffusionPipeline.from_pretrained(args.model_id)

pipe.to(torch_device="cuda", torch_dtype=torch.float16)
if not args.disable:
    pipe.unet = oneflow_compile(pipe.unet)

for _ in range(args.warmup):
    torch.manual_seed(args.seed)
    images = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
    ).images

torch.manual_seed(args.seed)
images = pipe(
    args.prompt, height=args.height, width=args.width, num_inference_steps=args.steps
).images
for i, image in enumerate(images):
    image.save(
        f"LCM-{args.width}x{args.height}-seed-{args.seed}-disable-{args.disable}.png"
    )
