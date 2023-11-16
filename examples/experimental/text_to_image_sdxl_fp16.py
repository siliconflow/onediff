"""
Compile to oneflow graph with :
oneflow_compile example: python examples/text_to_image_sdxl_fp16.py --compile
torch.compile example: python examples/text_to_image_sdxl_fp16.py
"""
import os
import argparse
from diffusers import StableDiffusionXLPipeline
import torch

from onediff.infer_compiler import torchbackend

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, default="/share_nfs/hf_models/stable-diffusion-xl-base-1.0"
)
parser.add_argument("--variant", type=str, default="fp16")
parser.add_argument(
    "--prompt",
    type=str,
    default="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
)
parser.add_argument(
    "--saved_image", type=str, required=False, default="xl-base-out.png"
)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--compile", type=(lambda x: str(x).lower() in ["true", "1", "yes"]), default=True
)
parser.add_argument("--graph", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

if args.compile:
    print("unet is compiled to oneflow.")
    if args.graph:
        print("unet is compiled to oneflow graph.")

torch.manual_seed(args.seed)

pipe = StableDiffusionXLPipeline.from_pretrained(
    args.model, torch_dtype=torch.float16, variant=args.variant, use_safetensors=True
)

if args.compile:
    if args.graph:
        os.environ["with_graph"] = "1"
        os.environ["ONEFLOW_MLIR_CSE"] = "1"
        os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
        os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
        os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
        os.environ["ONEFLOW_MLIR_FUSE_OPS_WITH_BACKWARD_IMPL"] = "1"
        os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
        os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"
        os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS"] = "1"
        os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"
        os.environ["ONEFLOW_KERNEL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"
        os.environ["ONEFLOW_KERNEL_CONV_ENABLE_CUTLASS_IMPL"] = "1"
        os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
        os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
        os.environ["ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT"] = "1"
    pipe.unet = torch.compile(
        pipe.unet, fullgraph=True, mode="reduce-overhead", backend=torchbackend
    )

pipe.to("cuda")

for i in range(3):
    image = pipe(
        prompt=args.prompt, height=768, width=768, num_inference_steps=50
    ).images[0]
    image.save(f"{i}-{args.saved_image}")
