# HF_HUB_OFFLINE=1 python3 examples/torch_interpretor.py
import torch
import torch._dynamo

# To force dynamo to capture attention
# torch._dynamo.allow_in_graph = lambda x : x
# torch._dynamo.allow_in_graph = torch._dynamo.disallow_in_graph
import diffusers
# import diffusers.models.attention_processor
# diffusers.models.attention_processor.AttnProcessor2_0 = diffusers.models.attention_processor.AttnProcessor

from diffusers import StableDiffusionPipeline
import os
import oneflow
import oneflow as flow
from torch.fx.experimental.proxy_tensor import make_fx
from torch.func import functionalize
import importlib
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import diffusers.utils.torch_utils
from onediff.infer_compiler import torchbackend

diffusers.utils.torch_utils.maybe_allow_in_graph = lambda x : x

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
)


def add_op_from_torch(node):
    getattr(oneflow, node["functioname"])(node.args)


torch._dynamo.config.verbose=True


# print(pipe.unet.state_dict().keys())
pipe.unet = torch.compile(pipe.unet, fullgraph=True, mode="reduce-overhead", backend=torchbackend)
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
# pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True, backend=my_custom_backend)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with torch.autocast("cuda"):
    images = pipe(prompt).images
    for i, image in enumerate(images):
        image.save(f"{prompt}-of-{i}.png")
