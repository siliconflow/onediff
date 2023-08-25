import os
import argparse
# cv2 must be imported before diffusers and oneflow to avlid error: AttributeError: module 'cv2.gapi' has no attribute 'wip'
# Maybe bacause oneflow use a lower version of cv2
import cv2
import oneflow as flow
import torch
# oneflow_compile should be imported before importing any diffusers
from onediff.infer_compiler import oneflow_compile 
from diffusers import StableDiffusionXLPipeline

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
parser.add_argument("--save", action=argparse.BooleanOptionalAction)
parser.add_argument("--load", action=argparse.BooleanOptionalAction)
parser.add_argument("--file", type=str, required=False, default="unet_compiled")
cmd_args = parser.parse_args()

# Normal SDXL
torch.manual_seed(cmd_args.seed)
pipe = StableDiffusionXLPipeline.from_pretrained(
    cmd_args.model, torch_dtype=torch.float16, variant=cmd_args.variant, use_safetensors=True
)
pipe.to("cuda")

# Compile unet with oneflow
if cmd_args.compile:
    pipe.unet = oneflow_compile(pipe.unet)
    print("unet is compiled to oneflow.")
    if cmd_args.load:
        # Load compiled unet with oneflow
        print("loading graphs...")
        pipe.unet._graph_load(cmd_args.file)

# Normal SDXL call
sizes = [1024, 896, 768]
for h in sizes:
    for w in sizes:
        for i in range(1):
            image = pipe(prompt=cmd_args.prompt, height=h, width=w, num_inference_steps=2).images[0]
            image.save(f"h{h}-w{w}-i{i}-{cmd_args.saved_image}")

# Save compiled unet with oneflow
if cmd_args.compile and cmd_args.save:
    print("saving graphs...")
    pipe.unet._graph_save(cmd_args.file)