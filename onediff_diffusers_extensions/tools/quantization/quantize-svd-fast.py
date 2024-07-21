import argparse
import time

import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image
from onediff.quantization import QuantPipeline
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, default="stabilityai/stable-video-diffusion-img2vid-xt"
)
parser.add_argument("--variant", type=str, default="fp16")
parser.add_argument("--quantized_model", type=str, required=True)
parser.add_argument("--input_image", type=str, required=True)
parser.add_argument("--steps", type=int, default=25)
parser.add_argument("--height", type=int, default=576)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument(
    "--conv_compute_density_threshold",
    type=int,
    default=900,
    help="The conv modules whose computational density is higher than the threshold will be quantized.",
)
parser.add_argument(
    "--linear_compute_density_threshold",
    type=int,
    default=500,
    help="The linear modules whose computational density is higher than the threshold will be quantized.",
)
parser.add_argument(
    "--conv_ssim_threshold",
    type=float,
    default=0,
    help="A similarity threshold that quantize convolution. The higher the threshold, the lower the accuracy loss caused by quantization.",
)
parser.add_argument(
    "--linear_ssim_threshold",
    type=float,
    default=0,
    help="A similarity threshold that quantize linear. The higher the threshold, the lower the accuracy loss caused by quantization.",
)
parser.add_argument(
    "--save_as_float",
    default=False,
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--cache_dir", type=str, default=None)
args = parser.parse_args()

pipe = QuantPipeline.from_pretrained(
    StableVideoDiffusionPipeline,
    args.model,
    torch_dtype=torch.float16,
    variant=args.variant,
    use_safetensors=True,
)
pipe.to("cuda")

input_image = load_image(args.input_image)
input_image.resize((args.width, args.height), Image.LANCZOS)
pipe_kwargs = dict(
    image=input_image,
    height=args.height,
    width=args.width,
    num_inference_steps=args.steps,
    decode_chunk_size=5,
    seed=args.seed,
)

start_time = time.time()

pipe.quantize(
    **pipe_kwargs,
    conv_compute_density_threshold=args.conv_compute_density_threshold,
    linear_compute_density_threshold=args.linear_compute_density_threshold,
    conv_ssim_threshold=args.conv_ssim_threshold,
    linear_ssim_threshold=args.linear_ssim_threshold,
    save_as_float=args.save_as_float,
    cache_dir=args.cache_dir,
)

pipe.save_quantized(args.quantized_model, safe_serialization=True)

end_time = time.time()

print(f"Quantize module time: {end_time - start_time}s")
