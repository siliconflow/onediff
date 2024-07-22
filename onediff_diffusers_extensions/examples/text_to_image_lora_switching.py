from collections import defaultdict, OrderedDict
from matplotlib import pyplot as plt
from pathlib import Path
import argparse

import torch
from onediff.utils.import_utils import is_oneflow_available, is_nexfort_available

USE_ONEFLOW = is_oneflow_available()
USE_NEXFORT = is_nexfort_available()
if USE_ONEFLOW:
    import oneflow as flow

from diffusers import StableDiffusionXLPipeline

from onediffx import compile_pipe
from onediffx.lora import load_and_fuse_lora, unfuse_lora

IMAGES = defaultdict(OrderedDict)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple demo of image generation.")
    parser.add_argument(
        "--prompt", type=str, default="a photo of an astronaut riding a horse on mars"
    )
    parser.add_argument(
        "--base", type=str, default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--loras", type=str, nargs="+", default=())
    cmd_args = parser.parse_args()
    return cmd_args


args = parse_args()

pipe = StableDiffusionXLPipeline.from_pretrained(
    args.base, variant="fp16", torch_dtype=torch.float16, safety_checker=None,
)
pipe = pipe.to("cuda")


# ---------- torch backend ----------
print("using torch backend")
for lora in args.loras:
    torch.manual_seed(args.seed)
    if Path(lora).exists():
        pipe.load_lora_weights(lora)
    else:
        lora, weight_name = lora.rsplit("/", 1)
        pipe.load_lora_weights(lora, weight_name=weight_name)

    pipe.fuse_lora()
    image = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
    ).images[0]
    image.save(f"torch-{args.prompt}-of-lora-{Path(lora).stem}.png")
    pipe.unfuse_lora(pipe)
    pipe.unload_lora_weights()
    IMAGES["torch"][Path(lora).stem] = image

# ---------- oneflow backend ----------
if USE_ONEFLOW:
    print("using oneflow backend")
    del pipe
    torch.cuda.empty_cache()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base, variant="fp16", torch_dtype=torch.float16, safety_checker=None,
    )
    pipe = pipe.to("cuda")
    pipe = compile_pipe(pipe, backend="oneflow")
    for _ in range(args.warmup):
        torch.manual_seed(args.seed)
        images = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
        ).images

    for lora in args.loras:
        if Path(lora).exists():
            load_and_fuse_lora(pipe, lora)
        else:
            lora, weight_name = lora.rsplit("/", 1)
            load_and_fuse_lora(pipe, lora, weight_name=weight_name)
        torch.manual_seed(args.seed)
        image = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
        ).images[0]
        image.save(f"oneflow-{args.prompt}-of-lora-{Path(lora).stem}.png")
        unfuse_lora(pipe)
        IMAGES["oneflow"][Path(lora).stem] = image

# ---------- nexfort backend ----------
if USE_NEXFORT:
    print("using nexfort backend")
    del pipe
    torch.cuda.empty_cache()
    if USE_ONEFLOW:
        flow.cuda.empty_cache()
    nexfort_options = {
        "mode": "cudagraphs:benchmark:max-autotune:low-precision:cache-all",
        "memory_format": "channels_last",
        "options": {
            "inductor.optimize_linear_epilogue": False,
            "overrides.conv_benchmark": True,
            "overrides.matmul_allow_tf32": True,
        },
    }
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base, variant="fp16", torch_dtype=torch.float16, safety_checker=None,
    )
    pipe = pipe.to("cuda")
    pipe = compile_pipe(pipe, backend="nexfort", options=nexfort_options)

    for _ in range(args.warmup):
        torch.manual_seed(args.seed)
        images = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
        ).images

    for lora in args.loras:
        torch.manual_seed(args.seed)
        if Path(lora).exists():
            load_and_fuse_lora(pipe, lora)
        else:
            lora, weight_name = lora.rsplit("/", 1)
            load_and_fuse_lora(pipe, lora, weight_name=weight_name)
        image = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
        ).images[0]
        image.save(f"nexfort-{args.prompt}-of-lora-{Path(lora).stem}.png")
        unfuse_lora(pipe)
        IMAGES["nexfort"][Path(lora).stem] = image


fig, axs = plt.subplots(3, 3, figsize=(10, 10))
for i, (backend, images) in enumerate(IMAGES.items()):
    for j, (lora, image) in enumerate(images.items()):
        axs[j, i].imshow(image)
        axs[j, i].axis("off")

column_titles = ["torch", "OneDiff (oneflow)", "OneDiff (nexfort)"]
for col in range(3):
    axs[0, col].set_title(column_titles[col])

plt.tight_layout(rect=[0.1, 0, 1, 1])
plt.savefig("result.png")
