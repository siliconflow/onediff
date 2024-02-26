import os
import argparse
import time

import torch
from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline
from onediffx import compile_pipe, compiler_config
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
)
parser.add_argument("--repo", type=str, default="ByteDance/SDXL-Lightning")
parser.add_argument("--cpkt", type=str, default="sdxl_lightning_4step_unet.safetensors")
parser.add_argument("--variant", type=str, default="fp16")
parser.add_argument(
    "--prompt",
    type=str,
    # default="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
    default="A girl smiling",
)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument(
    "--saved_image", type=str, required=False, default="sdxl-light-out.png"
)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--compile", type=(lambda x: str(x).lower() in ["true", "1", "yes"]), default=True,
)
args = parser.parse_args()

OUTPUT_TYPE = "pil"

n_steps = int(args.cpkt[len("sdxl_lightning_") : len("sdxl_lightning_") + 1])

is_lora_cpkt = "lora" in args.cpkt

if args.compile:
    from onediff.schedulers import EulerDiscreteScheduler
else:
    from diffusers import EulerDiscreteScheduler

if is_lora_cpkt:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    if os.path.isfile(os.path.join(args.repo, args.cpkt)):
        pipe.load_lora_weights(os.path.join(args.repo, args.cpkt))
    else:
        pipe.load_lora_weights(hf_hub_download(args.repo, args.cpkt))
    pipe.fuse_lora()
else:
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_config(args.base, subfolder="unet").to(
        "cuda", torch.float16
    )
    if os.path.isfile(os.path.join(args.repo, args.cpkt)):
        unet.load_state_dict(
            load_file(os.path.join(args.repo, args.cpkt), device="cuda")
        )
    else:
        unet.load_state_dict(
            load_file(hf_hub_download(args.repo, args.cpkt), device="cuda")
        )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base, unet=unet, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")

pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing"
)

# Compile the pipeline
if args.compile:
    pipe = compile_pipe(pipe,)

print("Warmup with running graphs...")
torch.manual_seed(args.seed)
image = pipe(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=n_steps,
    guidance_scale=0,
    output_type=OUTPUT_TYPE,
).images


# Normal run
print("Normal run...")
torch.manual_seed(args.seed)
start_t = time.time()
image = pipe(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=n_steps,
    guidance_scale=0,
    output_type=OUTPUT_TYPE,
).images

end_t = time.time()
print(f"e2e ({n_steps} steps) elapsed: {end_t - start_t} s")

image[0].save(args.saved_image)
