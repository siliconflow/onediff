# Compile and save to oneflow graph example: python examples/text_to_image_sdxl.py --compile --save
# Compile and load to oneflow graph example: python examples/text_to_image_sdxl.py --compile --load

import os
import argparse

import oneflow as flow
import torch

from onediff.infer_compiler import oneflow_compile
from diffusers import DiffusionPipeline

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
)
parser.add_argument(
    "--refiner", type=str, default="stabilityai/stable-diffusion-xl-refiner-1.0"
)
parser.add_argument("--with_refiner", action=argparse.BooleanOptionalAction)
parser.add_argument("--variant", type=str, default="fp16")
parser.add_argument(
    "--prompt",
    type=str,
    default="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
)
parser.add_argument("--n_steps", type=int, default=30)
parser.add_argument("--saved_image", type=str, required=False, default="sdxl-out.png")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--compile", type=(lambda x: str(x).lower() in ["true", "1", "yes"]), default=True
)
parser.add_argument("--num_dynamic_input_size", type=int, default=9)
parser.add_argument("--save", action=argparse.BooleanOptionalAction)
parser.add_argument("--load", action=argparse.BooleanOptionalAction)
parser.add_argument("--file", type=str, required=False, default="unet_compiled")
cmd_args = parser.parse_args()


# Normal SDXL pipeline init.
seed = torch.Generator("cuda").manual_seed(cmd_args.seed)
output_type = "pil"
# SDXL base: StableDiffusionXLPipeline
base = DiffusionPipeline.from_pretrained(
    cmd_args.base,
    torch_dtype=torch.float16,
    variant=cmd_args.variant,
    use_safetensors=True,
)
base.to("cuda")
if cmd_args.with_refiner:
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
    print("unet is compiled to oneflow.")
    base.unet = oneflow_compile(
        base.unet, options={"size": cmd_args.num_dynamic_input_size}
    )
    if cmd_args.with_refiner:
        refiner.unet = oneflow_compile(
            refiner.unet, options={"size": cmd_args.num_dynamic_input_size}
        )
        output_type = "latent"

# Load compiled unet with oneflow
if cmd_args.compile and cmd_args.load:
    print("loading graphs...")
    base.unet.warmup_with_load("base_" + cmd_args.file)
    if cmd_args.with_refiner:
        refiner.unet.warmup_with_load("refiner_" + cmd_args.file)

# Normal SDXL run
# sizes = [1024, 896, 768]
sizes = [1024, 896]
for h in sizes:
    for w in sizes:
        for i in range(2):
            image = base(
                prompt=cmd_args.prompt,
                height=h,
                width=w,
                generator=seed,
                num_inference_steps=cmd_args.n_steps,
                output_type=output_type,
            ).images
            if cmd_args.with_refiner:
                image = refiner(
                    prompt=cmd_args.prompt,
                    generator=seed,
                    num_inference_steps=cmd_args.n_steps,
                    image=image,
                ).images
            image[0].save(f"h{h}-w{w}-i{i}-{cmd_args.saved_image}")

# Save compiled unet with oneflow
if cmd_args.compile and cmd_args.save:
    print("saving graphs...")
    base.unet.save_graph("base_" + cmd_args.file)
    if cmd_args.with_refiner:
        refiner.unet.save_graph("refiner_" + cmd_args.file)
