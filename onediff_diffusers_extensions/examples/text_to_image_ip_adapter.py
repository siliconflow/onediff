import argparse
import json
import os
from pathlib import Path

import torch
from onediffx import compile_pipe, load_pipe, save_pipe

from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image

nexfort_options = {
    "mode": "cudagraphs:benchmark:max-autotune:low-precision:cache-all",
    "memory_format": "channels_last",
    "options": {
        "inductor.optimize_linear_epilogue": False,
        "overrides.conv_benchmark": True,
        "overrides.matmul_allow_tf32": True,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base", type=str, default="runwayml/stable-diffusion-v1-5"
)
parser.add_argument("--ipadapter", type=str, default="h94/IP-Adapter")
parser.add_argument("--subfolder", type=str, default="models")
parser.add_argument("--weight_name", type=str, default="ip-adapter_sd15.bin")
parser.add_argument("--scale", type=float, default=0.5)
parser.add_argument(
    "--input_image",
    type=str,
    default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner.png",
)
parser.add_argument(
    "--prompt",
    default="a polar bear sitting in a chair drinking a milkshake",
    help="Prompt",
)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--n_steps", type=int, default=30)
parser.add_argument(
    "--saved_image", type=str, required=False, default="ip-adapter-out.png"
)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--run", type=int, default=3)
parser.add_argument(
    "--compile",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=True,
)
parser.add_argument(
    "--compiler", type=str, default="oneflow", choices=["nexfort", "oneflow"]
)
parser.add_argument("--compile-options", type=str, default=nexfort_options)
parser.add_argument(
    "--cache-dir", default="./onediff_cache", help="Cache directory"
)
args = parser.parse_args()

# load an image
image = load_image(args.input_image)

# load stable diffusion and ip-adapter
pipe = AutoPipelineForText2Image.from_pretrained(
    args.base, torch_dtype=torch.float16
)
pipe.load_ip_adapter(
    args.ipadapter, subfolder=args.subfolder, weight_name=args.weight_name
)
pipe.set_ip_adapter_scale(args.scale)
pipe.to("cuda")


if args.compiler == "nexfort":
    compile_options = args.compile_options
    if isinstance(compile_options, str):
        compile_options = json.loads(compile_options)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "./.torchinductor")
else:
    compile_options = None

cache_path = os.path.join(args.cache_dir, type(pipe).__name__)

if args.compile:
    pipe = compile_pipe(pipe, backend=args.compiler, options=compile_options)
    if args.compiler == "oneflow" and os.path.exists(cache_path):
        load_pipe(pipe, cache_path)


# generate image
print("Warmup")
for i in range(args.warmup):
    images = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        ip_adapter_image=image,
        num_inference_steps=args.n_steps,
        generator=torch.manual_seed(args.seed),
    ).images

print("Run")
for i in range(args.run):
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        ip_adapter_image=image,
        num_inference_steps=args.n_steps,
        generator=torch.manual_seed(args.seed),
    ).images[0]
    image_path = (
        f"{Path(args.saved_image).stem}_{i}" + Path(args.saved_image).suffix
    )
    print(f"save output image to {image_path}")
    image.save(image_path)

if args.compiler == "oneflow":
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    save_pipe(pipe, cache_path)
