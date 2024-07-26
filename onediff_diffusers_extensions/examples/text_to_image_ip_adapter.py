import argparse
import json
import os
from pathlib import Path

import torch

from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
from onediffx import compile_pipe, load_pipe, save_pipe

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
parser.add_argument("--base", type=str, default="runwayml/stable-diffusion-v1-5")
parser.add_argument("--ipadapter", type=str, default="h94/IP-Adapter")
parser.add_argument("--subfolder", type=str, default="models")
parser.add_argument("--weight_name", type=str, default="ip-adapter_sd15.bin")
parser.add_argument("--scale", type=float, nargs="+", default=0.5)
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
parser.add_argument(
    "--negative-prompt",
    default="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
    help="Negative prompt",
)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--n_steps", type=int, default=100)
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
parser.add_argument("--cache-dir", default="./onediff_cache", help="cache directory")
parser.add_argument("--multi-resolution", action="store_true")
args = parser.parse_args()

# load an image
ip_adapter_image = load_image(args.input_image)

# load stable diffusion and ip-adapter
pipe = AutoPipelineForText2Image.from_pretrained(
    args.base,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.load_ip_adapter(
    args.ipadapter, subfolder=args.subfolder, weight_name=args.weight_name
)

# Set ipadapter scale as a tensor instead of a float
# If scale is a float, it cannot be modified after the graph is traced
ipadapter_scale = torch.tensor(0.6, dtype=torch.float, device="cuda")
pipe.set_ip_adapter_scale(ipadapter_scale)
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
        # TODO(WangYi): load pipe has bug here, which makes scale unchangeable
        # load_pipe(pipe, cache_path)
        pass


# generate image
print("Warmup")
for i in range(args.warmup):
    images = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        ip_adapter_image=ip_adapter_image,
        num_inference_steps=args.n_steps,
    ).images

print("Run")
scales = args.scale if isinstance(args.scale, list) else [args.scale]
for scale in scales:
    # Use ipadapter_scale.copy_ instead of pipeline.set_ip_adapter_scale to modify scale
    ipadapter_scale.copy_(torch.tensor(scale, dtype=torch.float, device="cuda"))
    pipe.set_ip_adapter_scale(ipadapter_scale)
    image = pipe(
        prompt=args.prompt,
        ip_adapter_image=ip_adapter_image,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.n_steps,
        generator=torch.Generator(device="cpu").manual_seed(0),
    ).images[0]
    image_path = (
        f"{Path(args.saved_image).stem}_{scale}" + Path(args.saved_image).suffix
    )
    print(f"save output image to {image_path}")
    image.save(image_path)

if args.multi_resolution:
    from itertools import product

    sizes = [1024, 512, 768, 256]
    for h, w in product(sizes, sizes):
        image = pipe(
            prompt=args.prompt,
            ip_adapter_image=ip_adapter_image,
            negative_prompt=args.negative_prompt,
            height=h,
            width=w,
            num_inference_steps=args.n_steps,
            generator=torch.Generator(device="cpu").manual_seed(0),
        ).images[0]
        print(f"Running at resolution: {h}x{w}")


if args.compiler == "oneflow":
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    save_pipe(pipe, cache_path)
