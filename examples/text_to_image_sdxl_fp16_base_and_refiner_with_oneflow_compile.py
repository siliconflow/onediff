import os
import argparse
# cv2 must be imported before diffusers and oneflow to avlid error: AttributeError: module 'cv2.gapi' has no attribute 'wip'
# Maybe bacause oneflow use a lower version of cv2
import cv2
import oneflow as flow
import torch
# oneflow_compile should be imported before importing any diffusers
from onediff.infer_compiler import oneflow_compile 
from diffusers import DiffusionPipeline

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base", type=str, default="/share_nfs/hf_models/stable-diffusion-xl-base-1.0"
)
parser.add_argument(
    "--refiner", type=str, default="stabilityai/stable-diffusion-xl-refiner-1.0"
)
parser.add_argument("--variant", type=str, default="fp16")
parser.add_argument(
    "--prompt",
    type=str,
    default="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
)
parser.add_argument("--n_steps", type=int, default=30)
parser.add_argument("--saved_image", type=str, required=False, default="xl-refiner-out.png")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--compile", action=argparse.BooleanOptionalAction)
parser.add_argument("--save", action=argparse.BooleanOptionalAction)
parser.add_argument("--load", action=argparse.BooleanOptionalAction)
parser.add_argument("--file", type=str, required=False, default="unet_compiled")
cmd_args = parser.parse_args()

# Normal SDXL
# seed = torch.manual_seed(cmd_args.seed)
generator = torch.Generator("cuda")
generator.manual_seed(cmd_args.seed)
seed = generator

# SDXL base: StableDiffusionXLPipeline
base = DiffusionPipeline.from_pretrained(
    cmd_args.base,
    torch_dtype=torch.float16,
    variant=cmd_args.variant,
    use_safetensors=True,
)
base.to("cuda")
# SDXL refiner: StableDiffusionXLImg2ImgPipeline
refiner = DiffusionPipeline.from_pretrained(
    cmd_args.refiner,
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant=cmd_args.variant,
)
refiner.to("cuda")

# Compile unet with oneflow
if cmd_args.compile:
    base.unet = oneflow_compile(base.unet)
    refiner.unet = oneflow_compile(refiner.unet)
    print("unet is compiled to oneflow.")
    if cmd_args.load:
        # Load compiled unet with oneflow
        print("loading graphs...")
        base.unet._graph_load("base_" + cmd_args.file)
        refiner.unet._graph_load("refiner_" + cmd_args.file)

# Normal SDXL call
sizes = [1024, 896, 768]
#sizes = [1024]
for h in sizes:
    for w in sizes:
        for i in range(3):
            image = base(
                prompt=cmd_args.prompt,
                height=h,
                width=w,
                generator=seed,
                num_inference_steps=cmd_args.n_steps,
                output_type="latent",
            ).images
            image = refiner(
                prompt=cmd_args.prompt,
                generator=seed,
                num_inference_steps=cmd_args.n_steps,
                image=image,
            ).images[0]
            image.save(f"h{h}-w{w}-i{i}-{cmd_args.saved_image}")

# Save compiled unet with oneflow
if cmd_args.compile and cmd_args.save:
    print("saving graphs...")
    base.unet._graph_save("base_" + cmd_args.file)
    refiner.unet._graph_save("refiner_" + cmd_args.file)
