# Compile and save to oneflow graph example: python examples/text_to_image_sdxl_save_load.py --save
# Compile and load to oneflow graph example: python examples/text_to_image_sdxl_save_load.py --load

import argparse
import os

import torch
import oneflow as flow  # usort: skip

from diffusers import DiffusionPipeline
from onediff.infer_compiler import oneflow_compile, OneflowCompileOptions

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
)
parser.add_argument("--variant", type=str, default="fp16")
parser.add_argument(
    "--prompt",
    type=str,
    default="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
)
parser.add_argument("--n_steps", type=int, default=3)
parser.add_argument("--saved_image", type=str, required=False, default="sdxl-out.png")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num_dynamic_input_size", type=int, default=9)
parser.add_argument("--save", action=argparse.BooleanOptionalAction)
parser.add_argument("--load", action=argparse.BooleanOptionalAction)
cmd_args = parser.parse_args()


# Normal SDXL pipeline init.
SEED = torch.Generator("cuda").manual_seed(cmd_args.seed)
OUTPUT_TYPE = "pil"
# SDXL base: StableDiffusionXLPipeline
base = DiffusionPipeline.from_pretrained(
    cmd_args.base,
    torch_dtype=torch.float16,
    variant=cmd_args.variant,
    use_safetensors=True,
)
base.to("cuda")

# To fix the bug of graph load of vae. Please refer to: https://github.com/siliconflow/onediff/issues/452
if base.vae.dtype == torch.float16 and base.vae.config.force_upcast:
    base.upcast_vae()

# Compile unet and vae
print("unet and vae is compiled to oneflow.")
compile_options = OneflowCompileOptions()
compile_options.max_cached_graph_size = cmd_args.num_dynamic_input_size

base.unet = oneflow_compile(base.unet, options=compile_options)
base.vae.decoder = oneflow_compile(base.vae.decoder, options=compile_options)
if base.text_encoder is not None:
    base.text_encoder = oneflow_compile(base.text_encoder, options=compile_options)
if base.text_encoder_2 is not None:
    base.text_encoder_2 = oneflow_compile(base.text_encoder_2, options=compile_options)

if cmd_args.load:
    print("Loading graphs to avoid compilation...")
    # run_warmup is True to run unet/vae once to make the cuda runtime ready.
    base.unet.load_graph("base_unet_compiled", run_warmup=True)
    base.vae.decoder.load_graph("base_vae_compiled", run_warmup=True)
    if base.text_encoder is not None:
        print("Loading text_encoder...")
        base.text_encoder.load_graph("base_text_encoder_compiled", run_warmup=True)
    if base.text_encoder_2 is not None:
        print("Loading text_encoder_2...")
        base.text_encoder_2.load_graph("base_text_encoder_2_compiled", run_warmup=True)
else:
    print("Warmup with running graphs...")
    sizes = [1024, 896]
    for h in sizes:
        for w in sizes:
            for i in range(2):
                image = base(
                    prompt=cmd_args.prompt,
                    height=h,
                    width=w,
                    generator=SEED,
                    num_inference_steps=cmd_args.n_steps,
                    output_type=OUTPUT_TYPE,
                ).images


# Normal SDXL run
print("Normal SDXL run...")
sizes = [1024, 896]
for h in sizes:
    for w in sizes:
        for i in range(2):
            image = base(
                prompt=cmd_args.prompt,
                height=h,
                width=w,
                generator=SEED,
                num_inference_steps=cmd_args.n_steps,
                output_type=OUTPUT_TYPE,
            ).images
            image[0].save(f"h{h}-w{w}-i{i}-{cmd_args.saved_image}")

# Save compiled graphs with oneflow
if cmd_args.save:
    print("Saving graphs...")
    base.unet.save_graph("base_unet_compiled")
    base.vae.decoder.save_graph("base_vae_compiled")
    if base.text_encoder is not None:
        base.text_encoder.save_graph("base_text_encoder_compiled")
    if base.text_encoder_2 is not None:
        base.text_encoder_2.save_graph("base_text_encoder_2_compiled")
