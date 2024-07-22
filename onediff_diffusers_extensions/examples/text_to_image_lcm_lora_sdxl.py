import argparse
import importlib.metadata
import os

from packaging import version


def parse_args():
    parser = argparse.ArgumentParser(description="Simple demo of LCM image generation.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Model id or local path to the Stable Diffusion base model.",
    )
    parser.add_argument(
        "--adapter_id",
        type=str,
        default="latent-consistency/lcm-lora-sdxl",
        help="Model id or local path to the LCM Lora adapter model.",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup option sets the number of preliminary runs of the pipeline. These initial runs are necessary to ensure accurate performance data during testing.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--disable", action="store_true", help="Disable Onediff speeding up."
    )

    parser.add_argument(
        "--mlir_enable_inference_optimization",
        action="store_true",
        help="If this option is enabled, it will trigger additional optimizations, but it may potentially affect the quality of the output image.",
    )
    parser.add_argument(
        "--output", type=str, default="", help="Output image file name."
    )

    args = parser.parse_args()
    return args


from diffusers import AutoPipelineForText2Image, LCMScheduler

args = parse_args()

if not args.disable:
    from onediff.infer_compiler import oneflow_compile
import torch

pipe = AutoPipelineForText2Image.from_pretrained(
    args.model_id, torch_dtype=torch.float16, variant="fp16"
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

if not args.disable:
    pipe.unet = oneflow_compile(pipe.unet)


if not args.mlir_enable_inference_optimization:
    os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "0"
else:
    os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"

pipe.load_lora_weights(args.adapter_id)
if hasattr(pipe, "fuse_lora"):
    pipe.fuse_lora()
else:
    print(f"pipe.fuse_lora doesn't exist, exit")
    exit(1)

for _ in range(args.warmup):
    torch.manual_seed(args.seed)
    images = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
    ).images

torch.manual_seed(args.seed)
images = pipe(
    args.prompt, height=args.height, width=args.width, num_inference_steps=args.steps
).images
for i, image in enumerate(images):
    if args.output == "":
        image.save(
            f"LCM-LoRA-{args.width}x{args.height}-seed-{args.seed}-disable-{args.disable}-mlir-{args.mlir_enable_inference_optimization}.png"
        )
    else:
        image.save(f"{args.output}.png")
