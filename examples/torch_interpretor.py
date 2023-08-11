from diffusers import StableDiffusionPipeline

import torch
import oneflow

from unet_torch_interplay import MockCtx
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
)


def add_op_from_torch(node):
    getattr(oneflow, node["functioname"])(node.args)


torch._dynamo.config.verbose=True

def replace_cls(obj):
    cls = type(obj)
    if cls.__module__.startswith("torch"):
        mod_name = cls.__module__.replace("torch", "oneflow")
        mod = globals()[mod_name]
        cls = getattr(mod, cls.__name__)
        return cls(str(obj))
    else:
        return obj

class OneFlowInterpreter(torch.fx.Interpreter):
    from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate
    from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
    def call_function(self, target : Target,
                      args : Tuple, kwargs : Dict) -> Any:
        if target == torch.sigmoid:
            return torch.neg(*args, **kwargs)
        return super().call_function(target, args, kwargs)

    def call_method(self, target : Target,
                    args : Tuple, kwargs : Dict) -> Any:
        if target == 'neg':
            call_self, *args_tail = args
            return call_self.sigmoid(*args_tail, **kwargs)
        print([type(a) for a in args], [type(v) for v in kwargs.values()])
        args = [replace_cls(a) for a in args]
        print("after")
        print([str(a) for a in args])
        print([type(a) for a in args], [type(v) for v in kwargs.values()])
        print([(key, str(value)) for key, value in kwargs.items()])
        return super().call_method(target, args, kwargs)

def torchbackend(gm, example_inputs):
    import oneflow as flow
    # TODO: when initialzing oneflow variables, find them in the state dict and reuse them using dlpack
    # gm.print_readable()
    def wrapped_forward(*args, **kwargs):
        import oneflow as flow
        args = [flow.utils.tensor.from_torch(a) for a in args]
        print([type(a) for a in args], [type(v) for v in kwargs.values()])
        # with MockCtx():
        return OneFlowInterpreter(gm).run(*args, **kwargs)
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
