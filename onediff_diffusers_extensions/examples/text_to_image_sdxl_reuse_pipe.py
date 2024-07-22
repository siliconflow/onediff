import argparse
import os

import torch
from diffusers import StableDiffusionXLPipeline

from onediff.infer_compiler import oneflow_compile

# import diffusers
# diffusers.logging.set_verbosity_info()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
)
parser.add_argument(
    "--new_base",
    type=str,
    default="dataautogpt3/OpenDalleV1.1",
)
parser.add_argument("--variant", type=str, default="fp16")
parser.add_argument(
    "--prompt",
    type=str,
    default="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--n_steps", type=int, default=30)
parser.add_argument("--guidance_scale", type=float, default=7.5)
parser.add_argument("--saved_image", type=str, required=False, default="sdxl-out.png")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--compile_unet",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=True,
)
parser.add_argument(
    "--compile_vae",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=True,
)
parser.add_argument(
    "--run_multiple_resolutions",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=True,
)
args = parser.parse_args()

# Normal SDXL pipeline init.
OUTPUT_TYPE = "pil"

# SDXL base: StableDiffusionXLPipeline
base = StableDiffusionXLPipeline.from_pretrained(
    args.base,
    torch_dtype=torch.float16,
    variant=args.variant,
    use_safetensors=True,
)
base.to("cuda")

# Compile unet with oneflow
if args.compile_unet:
    print("Compiling unet with oneflow.")
    compiled_unet = oneflow_compile(base.unet)
    base.unet = compiled_unet

# Compile vae with oneflow
if args.compile_vae:
    print("Compiling vae with oneflow.")
    compiled_decoder = oneflow_compile(base.vae.decoder)
    base.vae.decoder = compiled_decoder

# Warmup with run
# Will do compilatioin in the first run
print("Warmup with running graphs...")
torch.manual_seed(args.seed)
image = base(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.n_steps,
    generator=torch.manual_seed(0),
    output_type=OUTPUT_TYPE,
    guidance_scale=args.guidance_scale,
).images
del base

torch.cuda.empty_cache()

print("loading new base")
if str(args.new_base).endswith(".safetensors"):
    new_base = StableDiffusionXLPipeline.from_single_file(
        args.new_base,
        torch_dtype=torch.float16,
        variant=args.variant,
        use_safetensors=True,
    )
else:
    new_base = StableDiffusionXLPipeline.from_pretrained(
        args.new_base,
        torch_dtype=torch.float16,
        variant=args.variant,
        use_safetensors=True,
    )
new_base.to("cuda")

print("New base running by torch backend")
torch.manual_seed(args.seed)
image = new_base(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.n_steps,
    generator=torch.manual_seed(0),
    output_type=OUTPUT_TYPE,
    guidance_scale=args.guidance_scale,
).images
image[0].save(f"new_base_without_graph_h{args.height}-w{args.width}-{args.saved_image}")
image_eager = image[0]


# Update the unet and vae
# load_state_dict(state_dict, strict=True, assign=False), assign is False means copying them inplace into the module’s current parameters and buffers.
# Reference: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict
print("Loading state_dict of new base into compiled graph")
compiled_unet._torch_module.load_state_dict(new_base.unet.state_dict())
compiled_decoder._torch_module.load_state_dict(new_base.vae.decoder.state_dict())

new_base.unet = compiled_unet
new_base.vae.decoder = compiled_decoder

torch.cuda.empty_cache()

# Normal SDXL run
print("Re-use the compiled graph")
torch.manual_seed(args.seed)
image = new_base(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.n_steps,
    generator=torch.manual_seed(0),
    output_type=OUTPUT_TYPE,
    guidance_scale=args.guidance_scale,
).images
image[0].save(f"new_base_reuse_graph_h{args.height}-w{args.width}-{args.saved_image}")
image_graph = image[0]

import numpy as np
from skimage.metrics import structural_similarity

ssim = structural_similarity(
    np.array(image_eager), np.array(image_graph), channel_axis=-1, data_range=255
)
print(f"ssim between naive torch and re-used graph is {ssim}")


# Should have no compilation for these new input shape
print("Test run with multiple resolutions...")
if args.run_multiple_resolutions:
    sizes = [960, 720, 896, 768]
    if "CI" in os.environ:
        sizes = [360]
    for h in sizes:
        for w in sizes:
            image = new_base(
                prompt=args.prompt,
                height=h,
                width=w,
                num_inference_steps=args.n_steps,
                generator=torch.manual_seed(0),
                output_type=OUTPUT_TYPE,
            ).images


# print("Test run with other another uncommon resolution...")
# if args.run_multiple_resolutions:
#     h = 544
#     w = 408
#     image = base(
#         prompt=args.prompt,
#         height=h,
#         width=w,
#         num_inference_steps=args.n_steps,
#         output_type=OUTPUT_TYPE,
#     ).images
