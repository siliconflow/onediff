from diffusers import StableDiffusionPipeline

import torch
import oneflow
import oneflow as flow
import importlib
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
)


def add_op_from_torch(node):
    getattr(oneflow, node["functioname"])(node.args)


torch._dynamo.config.verbose=True

def replace_class(cls):
    if cls.__module__.startswith("torch"):
        mod_name = cls.__module__.replace("torch", "oneflow")
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls.__name__)

def replace_obj(obj):
    cls = type(obj)
    if cls == torch.dtype:
        return {
            "torch.float16": flow.float16,
            "torch.float32": flow.float32,
            "torch.double": flow.double,
            "torch.int8": flow.int8,
            "torch.int32": flow.int32,
            "torch.int64": flow.int64,
            "torch.uint8": flow.uint8,
        }[str(obj)]
    if cls == torch.fx.immutable_collections.immutable_list:
        return [e for e in obj]
    replacement = replace_class(cls)
    if replacement is not None:
        return replacement(str(obj))
    else:
        return obj

def replace_func(func):
    if func.__module__.startswith("torch"):
        mod_name = func.__module__.replace("torch", "oneflow")
        mod = globals()[mod_name]
        return getattr(mod, func.__name__)
    else:
        return func

def map_args(args, kwargs):
    args = [replace_obj(a) for a in args]
    kwargs = dict((k, replace_obj(v)) for (k, v) in kwargs.items())
    return (args, kwargs)

def print_types(args, kwargs):
    print([type(a) for a in args], [type(v) for v in kwargs.values()])

class ProxySubmodule:
    def __init__(self, submod):
        self.submod = submod
        attrnames = [k for k in submod.__dict__.keys() if not k.startswith("_")]
        print(f"{attrnames=}")

    def __getattribute__(self, attribute):
        submod = object.__getattribute__(self, "submod")
        if attribute == "submod":
            return submod
        elif attribute in ["forward"]:
            replacement = replace_class(type(self.submod))
            return getattr(replacement, attribute)
        else:
            return getattr(submod, attribute)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        replacement = replace_class(type(self.submod))
        return replacement.__call__(self, *args, **kwargs)

class OneFlowInterpreter(torch.fx.Interpreter):
    from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate

    def run_node(self, n : Node) -> Any:
        print("\nrun", n)
        return super().run_node(n)

    def call_function(self, target : Target,
                      args : Tuple, kwargs : Dict) -> Any:
        if target == torch.sigmoid:
            return torch.neg(*args, **kwargs)
        print_types(args, kwargs)
        args, kwargs = map_args(args, kwargs)
        print_types(args, kwargs)
        target = replace_func(target)
        return super().call_function(target, args, kwargs)

    def call_method(self, target : Target,
                    args : Tuple, kwargs : Dict) -> Any:
        if target == 'neg':
            call_self, *args_tail = args
            return call_self.sigmoid(*args_tail, **kwargs)
        args, kwargs = map_args(args, kwargs)
        print_types(args, kwargs)
        return super().call_method(target, args, kwargs)

    def call_module(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        submod = self.fetch_attr(target)
        print(f"{type(submod)=}")
        for name, param in submod.named_parameters():
            print(name, param)
        submod = ProxySubmodule(submod)
        return submod(*args, **kwargs)

def torchbackend(gm, example_inputs):
    import oneflow as flow
    # TODO: when initialzing oneflow variables, find them in the state dict and reuse them using dlpack
    # gm.print_readable()
    def wrapped_forward(*args, **kwargs):
        import oneflow as flow
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
