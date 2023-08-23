import os
import argparse
# cv2 must be imported before diffusers and oneflow to avlid error: AttributeError: module 'cv2.gapi' has no attribute 'wip'
# Maybe bacause oneflow use a lower version of cv2
import cv2
import oneflow as flow
# obj_1f_from_torch should be import before import any diffusers
from onediff.infer_compiler import obj_1f_from_torch

from diffusers import StableDiffusionXLPipeline
import torch
#from onediff.infer_compiler import torchbackend
from onediff.infer_compiler.with_fx_graph import _get_of_module

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
parser.add_argument("--saved_image", type=str, required=False, default="xl-base-out.png")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--compile", action=argparse.BooleanOptionalAction)
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
pipe.to("cuda")

if args.compile:
    if args.graph:
        os.environ["with_graph"] = "1"
        os.environ["ONEFLOW_MLIR_CSE"] = "1"
        os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
        os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
        os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
        os.environ["ONEFLOW_MLIR_FUSE_OPS_WITH_BACKWARD_IMPL"] = "1"
        # Open this will raise error
        # os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
        os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"
        os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS"] = "1"
        os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"
        os.environ["ONEFLOW_KERNEL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"
        os.environ["ONEFLOW_KERNEL_CONV_ENABLE_CUTLASS_IMPL"] = "1"
        os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
        os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
        os.environ["ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT"] = "1"
    torch2flow = {}

    def get_deployable(of_md):
        from oneflow.framework.args_tree import ArgsTree
        def input_fn(value):
            if isinstance(value, torch.Tensor):
                return flow.utils.tensor.from_torch(value)
            else:
                return value

        def output_fn(value):
            if isinstance(value, flow.Tensor):
                return flow.utils.tensor.to_torch(value)
            else:
                return value

        class DeplayableModule(of_md.__class__):
            def __call__(self, *args, **kwargs):
                args_tree = ArgsTree((args, kwargs), False, tensor_type=torch.Tensor)
                out = args_tree.map_leaf(input_fn)
                mapped_args = out[0]
                mapped_kwargs = out[1]

                output = super().__call__(*mapped_args, **mapped_kwargs)

                out_tree = ArgsTree((output, None), False)
                out = out_tree.map_leaf(output_fn)
                return out[0]

        of_md.__class__ = DeplayableModule
        return of_md

    unet = _get_of_module(pipe.unet, torch2flow)
    d_unet = get_deployable(unet)
    print(type(unet))
    pipe.unet = d_unet

for i in range(3):
    image = pipe(prompt=args.prompt).images[0]
    image.save(f"{i}-{args.saved_image}")
