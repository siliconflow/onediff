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
from attention_1f import BasicTransformerBlock
from attention_processor_1f import Attention
from backend_1f import ProxySubmodule, OneFlowInterpreter

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

def torchbackend(gm, example_inputs):
    # TODO: when initialzing oneflow variables, find them in the state dict and reuse them using dlpack
    # gm.print_readable()
    def wrapped_forward(*args, **kwargs):
        for fn in [diffusers.models.attention.BasicTransformerBlock]:
            torch._dynamo.allowed_functions._allowed_function_ids.remove(id(fn))
        # gmf = make_fx(functionalize(gm))(*args, **kwargs)
        # gmf.print_readable()
        args = [flow.utils.tensor.from_torch(a) for a in args]
        print([type(a) for a in args], [type(v) for v in kwargs.values()])
        # with MockCtx():
        return OneFlowInterpreter(gm, garbage_collect_values=False).run(*args, **kwargs)
    return wrapped_forward

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
