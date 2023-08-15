import argparse
from diffusers import StableDiffusionXLPipeline
import torch

from onediff.infer_compiler import torchbackend


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, default="/ssd/home/chenhoujiang/stable-diffusion-xl-base-1.0"
)
parser.add_argument("--variant", type=str, default="fp16")
parser.add_argument(
    "--prompt",
    type=str,
    default="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
)
parser.add_argument("--saved_image", type=str, required=True)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

torch.manual_seed(args.seed)

pipe = StableDiffusionXLPipeline.from_pretrained(
    args.model, torch_dtype=torch.float16, variant=args.variant, use_safetensors=True
)

pipe.unet = torch.compile(pipe.unet, fullgraph=True, mode="reduce-overhead", backend=torchbackend)

pipe.to("cuda")

image = pipe(prompt=args.prompt).images[0]
image.save(args.saved_image)
