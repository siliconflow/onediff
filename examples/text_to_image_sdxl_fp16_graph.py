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
from onediff.infer_compiler.with_fx_graph import _get_of_module, UNetGraph

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
cmd_args = parser.parse_args()

if cmd_args.compile:
    print("unet is compiled to oneflow.")
    if cmd_args.graph:
        print("unet is compiled to oneflow graph.")

torch.manual_seed(cmd_args.seed)

pipe = StableDiffusionXLPipeline.from_pretrained(
    cmd_args.model, torch_dtype=torch.float16, variant=cmd_args.variant, use_safetensors=True
)
pipe.to("cuda")

if cmd_args.compile:
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

        if cmd_args.graph:
            unet_graph = UNetGraph(of_md)

        class DeplayableModule(of_md.__class__):
            def __call__(self, *args, **kwargs):
                args_tree = ArgsTree((args, kwargs), False, tensor_type=torch.Tensor)
                out = args_tree.map_leaf(input_fn)
                mapped_args = out[0]
                mapped_kwargs = out[1]

                if cmd_args.graph:
                    output = unet_graph(*mapped_args, **mapped_kwargs)
                else:
                    output = super().__call__(*mapped_args, **mapped_kwargs)


                out_tree = ArgsTree((output, None), False)
                out = out_tree.map_leaf(output_fn)
                return out[0]

        of_md.__class__ = DeplayableModule
        return of_md

    torch2flow = {}
    unet = _get_of_module(pipe.unet, torch2flow)
    d_unet = get_deployable(unet)
    pipe.unet = d_unet

sizes = [1024, 896, 768]
for h in sizes:
    for w in sizes:
        for i in range(2):
            image = pipe(prompt=cmd_args.prompt, height=h, width=w, num_inference_steps=50).images[0]
            image.save(f"h{h}-w{w}-i{i}-{cmd_args.saved_image}")
