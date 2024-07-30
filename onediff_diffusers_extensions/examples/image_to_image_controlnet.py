import argparse

import cv2
import numpy as np
import torch

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
)

from diffusers.utils import load_image
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str, default="stabilityai/sd-turbo")
parser.add_argument(
    "--controlnet", type=str, default="thibaud/controlnet-sd21-canny-diffusers"
)
parser.add_argument(
    "--input_image",
    type=str,
    default="https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
)
parser.add_argument(
    "--prompt",
    type=str,
    default="chinese painting style women",
)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--n_steps", type=int, default=7)
parser.add_argument(
    "--saved_image", type=str, required=False, default="i2i_controlnet-out.png"
)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--run", type=int, default=3)
parser.add_argument(
    "--compile_unet",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=True,
)
parser.add_argument(
    "--quant_unet",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=True,
)
parser.add_argument(
    "--compile_vae",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=True,
)
parser.add_argument(
    "--compile_ctrlnet",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=True,
)
args = parser.parse_args()

# load an image
image = load_image(args.input_image)
image = np.array(image)

# get canny image
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

# load control net and stable diffusion
# reference: https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet
controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    args.base, controlnet=controlnet, torch_dtype=torch.float16
)

# speed up diffusion process with faster scheduler
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

if args.compile_unet:
    from onediff.infer_compiler import oneflow_compile

    if args.quant_unet:
        from onediff.optimization.quant_optimizer import quantize_model

        pipe.unet = quantize_model(pipe.unet, inplace=True)
    pipe.unet = oneflow_compile(pipe.unet)
    torch.cuda.empty_cache()
if args.compile_vae:
    from onediff.infer_compiler import oneflow_compile

    # ImageToImage has an encoder and decoder, so we need to compile them separately.
    pipe.vae.encoder = oneflow_compile(pipe.vae.encoder)
    pipe.vae.decoder = oneflow_compile(pipe.vae.decoder)

if args.compile_ctrlnet:
    from onediff.infer_compiler import oneflow_compile

    pipe.controlnet = oneflow_compile(pipe.controlnet)


# generate image
generator = torch.manual_seed(args.seed)

print("Warmup")
for i in range(args.warmup):
    images = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.n_steps,
        generator=generator,
        image=image,
        control_image=canny_image,
    ).images

print("Run")
import time

from tqdm import tqdm

for i in tqdm(range(args.run), desc="Pipe processing", unit="i"):
    start_t = time.time()
    image = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.n_steps,
        generator=generator,
        image=image,
        control_image=canny_image,
    ).images[0]
    torch.cuda.synchronize()
    end_t = time.time()
    print(f"e2e {i} ) elapsed: {end_t - start_t} s")

image.save(f"{i=}th_{args.saved_image}.png")
