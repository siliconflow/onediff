import os
import argparse
from diffusers import StableDiffusionXLPipeline
import torch

from onediff.infer_compiler import oneflow_backend


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, default="/share_nfs/hf_models/stable-diffusion-xl-base-1.0"
)
parser.add_argument("--variant", type=str, default="fp16")
parser.add_argument(
    "--prompt",
    type=str,
    default="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
)
parser.add_argument("--saved_image", type=str, required=False, default="xl-base-out.png")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--compile", action=argparse.BooleanOptionalAction)
parser.add_argument("--graph", action=argparse.BooleanOptionalAction)
parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

if args.compile:
    print("unet is compiled to oneflow.")
    if args.graph:
        os.environ["with_graph"] = "1"
        print("unet is compiled to oneflow graph.")
        if args.dynamic:
            os.environ["dynamic_shape"] = "1"
            print("unet is compiled to oneflow dynamic shape graph.")

torch.manual_seed(args.seed)

# Reference: https://huggingface.co/docs/diffusers/v0.20.0/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    args.model, torch_dtype=torch.float16, variant=args.variant, use_safetensors=True
)

# Reference: https://pytorch.org/docs/stable/generated/torch.compile.html
# dynamic=True will cause error in torch dynamo trace
pipe.unet = torch.compile(pipe.unet, disable=(not args.compile), dynamic=False, fullgraph=True, mode="default", backend=oneflow_backend)

pipe.to("cuda")

# sizes = [1024, 896, 768]
sizes = [896, 768]
for h in sizes:
    for w in sizes:
        for i in range(2):
            image = pipe(prompt=args.prompt, height=h, width=w, num_inference_steps=50).images[0]
            image.save(f"h{h}-w{w}-i{i}-{args.saved_image}")
