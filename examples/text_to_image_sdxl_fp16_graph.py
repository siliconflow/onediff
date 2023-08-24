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
parser.add_argument("--file", type=str, required=False, default="deployable_unet")
parser.add_argument("--save", action=argparse.BooleanOptionalAction)
parser.add_argument("--load", action=argparse.BooleanOptionalAction)
cmd_args = parser.parse_args()

# For compile with oneflow
if cmd_args.compile:
    print("unet is compiled to oneflow.")
    if cmd_args.graph:
        print("unet is compiled to oneflow graph.")
    
def get_deployable(torch_md):
    torch2flow = {}
    of_md = _get_of_module(torch_md, torch2flow)
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
        dpl_graph = UNetGraph(of_md)

    class DeplayableModule(of_md.__class__):
        def __call__(self, *args, **kwargs):
            args_tree = ArgsTree((args, kwargs), False, tensor_type=torch.Tensor)
            out = args_tree.map_leaf(input_fn)
            mapped_args = out[0]
            mapped_kwargs = out[1]

            if cmd_args.graph:
                output = self._dpl_graph(*mapped_args, **mapped_kwargs)
            else:
                output = super().__call__(*mapped_args, **mapped_kwargs)


            out_tree = ArgsTree((output, None), False)
            out = out_tree.map_leaf(output_fn)
            return out[0]
        
        def _dpl_load(self, file_path):
            self._dpl_graph.warmup_with_load(file_path)
        
        def _dpl_save(self, file_path):
            self._dpl_graph.save_graph(file_path)

    of_md.__class__ = DeplayableModule
    if cmd_args.graph:
        of_md._dpl_graph = dpl_graph
        if cmd_args.load:
            print("loading deployable graphs...")
            of_md._dpl_load(cmd_args.file)
    return of_md

# Normal SDXL
torch.manual_seed(cmd_args.seed)
pipe = StableDiffusionXLPipeline.from_pretrained(
    cmd_args.model, torch_dtype=torch.float16, variant=cmd_args.variant, use_safetensors=True
)
pipe.to("cuda")

# Compile unet with oneflow
if cmd_args.compile:
    pipe.unet = get_deployable(pipe.unet)

# Normal SDXL call
sizes = [1024, 896, 768]
for h in sizes:
    for w in sizes:
        for i in range(2):
            image = pipe(prompt=cmd_args.prompt, height=h, width=w, num_inference_steps=50).images[0]
            image.save(f"h{h}-w{w}-i{i}-{cmd_args.saved_image}")

# Save compiled unet with oneflow
if cmd_args.save:
    print("saving deployable graphs...")
    pipe.unet._dpl_save(cmd_args.file)