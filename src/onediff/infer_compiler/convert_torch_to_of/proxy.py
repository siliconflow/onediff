import os
import torch
import oneflow as flow
import importlib
import logging


logger = logging.getLogger(__name__)
from functools import singledispatch
from onediff.infer_compiler.import_tools import get_classes_in_package, print_green
from onediff.infer_compiler.import_tools.finder import get_mock_cls_name

__all__ = [
    "replace_class",
    "replace_obj",
    "replace_func",
    "map_args",
    "ProxySubmodule",
]

from .globals import PROXY_OF_MDS as __of_mds


import diffusers
from typing import Any
from .attention_processor_1f import Attention

_is_diffusers_quant_available = False
try:
    import diffusers_quant

    _is_diffusers_quant_available = True
except:
    pass


def replace_class(cls):
    global __of_mds

    if cls.__module__.startswith("torch"):
        mod_name = cls.__module__.replace("torch", "oneflow")
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls.__name__)
    # TODO https://github.com/Oneflow-Inc/oneflow/issues/10328
    elif cls == diffusers.models.attention_processor.Attention:
        return Attention

    # full_cls_name = str(cls.__module__) + "." + str(cls.__name__)
    full_cls_name = get_mock_cls_name(str(cls.__module__) + "." + str(cls.__name__))
    if full_cls_name in __of_mds:
        return __of_mds[full_cls_name]

    if _is_diffusers_quant_available:
        if cls == diffusers_quant.FakeQuantModule:
            return diffusers_quant.OneFlowFakeQuantModule
        if cls == diffusers_quant.StaticQuantConvModule:
            return diffusers_quant.OneFlowStaticQuantConvModule
        if cls == diffusers_quant.DynamicQuantConvModule:
            return diffusers_quant.OneFlowDynamicQuantConvModule
        if cls == diffusers_quant.StaticQuantLinearModule:
            return diffusers_quant.OneFlowStaticQuantLinearModule
        if cls == diffusers_quant.DynamicQuantLinearModule:
            return diffusers_quant.OneFlowDynamicLinearQuantModule
        
    raise RuntimeError("can't find oneflow module for: " + str(cls))

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
        if cls in [torch.device]:
            return replacement(str(obj))
        elif cls == torch.nn.parameter.Parameter:
            return flow.utils.tensor.from_torch(obj.data)
        else:
            raise RuntimeError("don't know how to create oneflow obj for: " + str(cls))
    else:
        return obj


def replace_func(func):
    if func == torch.conv2d:
        return flow.nn.functional.conv2d
    if func == torch._C._nn.linear:
        return flow.nn.functional.linear
    if func.__module__.startswith("torch"):
        mod_name = func.__module__.replace("torch", "oneflow")
        mod = importlib.import_module(mod_name)
        return getattr(mod, func.__name__)
    else:
        return func

def map_args(args, kwargs):
    args = [replace_obj(a) for a in args]
    kwargs = dict((k, replace_obj(v)) for (k, v) in kwargs.items())
    return (args, kwargs)




class ProxySubmodule:
    def __init__(self, submod):
        self._1f_proxy_submod = submod
        self._1f_proxy_parameters = dict()
        self._1f_proxy_children = dict()
    

    def __getitem__(self, index): # __getitem__
        from collections.abc import Iterable
        if isinstance(self._1f_proxy_submod,Iterable):
            submod = self._1f_proxy_submod[index]
            from .register import torch2of
            return torch2of(submod)
        else:
            raise RuntimeError("can't getitem for: " + str(type(self._1f_proxy_submod)))
        
    def __repr__(self) -> str:
        return self._1f_proxy_submod.__repr__() + " 1f_proxy"

    def __getattribute__(self, attribute):
        # import os 
        # if os.environ.get('DEBUG_MODEL','-1') == '1':
        #     print('DEBUG: checkpointing disabled')
        #     import pdb; pdb.set_trace()

        if attribute.startswith("_1f_proxy"):
            return object.__getattribute__(self, attribute)
        elif attribute in ["forward", "_conv_forward"]:
            from .register import torch2of
            of_mod  =  torch2of(self._1f_proxy_submod)
            return getattr(of_mod, attribute)
            # replacement = replace_class(type(self._1f_proxy_submod))
            # import os 
            # if os.environ.get('DEBUG_MODEL','-1') == '1':
            #     print('DEBUG: checkpointing disabled')
            #     import pdb; pdb.set_trace()
            # return lambda *args, **kwargs: getattr(replacement, attribute)(
            #     self, *args, **kwargs
            # )
        elif (
            isinstance(
                self._1f_proxy_submod, diffusers.models.attention_processor.Attention
            )
            and attribute == "get_attention_scores"
        ):
            replacement = replace_class(type(self._1f_proxy_submod))
            return lambda *args, **kwargs: getattr(replacement, attribute)(
                self, *args, **kwargs
            )
        elif (
            isinstance(self._1f_proxy_submod, torch.nn.Linear)
            and attribute == "use_fused_matmul_bias"
        ):
            return (
                self.bias is not None
                and os.getenv("ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR") == "1"
            )
        elif (
            isinstance(self._1f_proxy_submod, torch.nn.Dropout)
            and attribute == "generator"
        ):
            return flow.Generator()
        elif (
            isinstance(self._1f_proxy_submod, torch.nn.Conv2d)
            and attribute == "channel_pos"
        ):
            return "channels_first"
        else:
            a = getattr(self._1f_proxy_submod, attribute)
            # isinstance(getattr(self._1f_proxy_submod, attribute), torch.nn.Module)
            if isinstance(a, torch.Tensor):
                a = flow.utils.tensor.from_torch(a.data)
            elif isinstance(a, torch.nn.parameter.Parameter):
                # TODO(oneflow): assert a.requires_grad == False
                if attribute not in self._1f_proxy_parameters:
                    a = flow.utils.tensor.from_torch(a.data)
                    self._1f_proxy_parameters[attribute] = a
                else:
                    a = self._1f_proxy_parameters[attribute]
            elif isinstance(a, torch.nn.ModuleList):
                a = [ProxySubmodule(m) for m in a]
            elif isinstance(a, torch.nn.modules.container.Sequential):
                from .register import torch2of
                a = flow.nn.Sequential(*[torch2of(m) for m in a])
                return a
            elif isinstance(a, torch.nn.Module):
                if attribute not in self._1f_proxy_children:
                    # a = ProxySubmodule(a)
                    from .register import torch2of
                    a = torch2of(a)
                    self._1f_proxy_children[attribute] = a
                else:
                    a = self._1f_proxy_children[attribute]

            full_name = ".".join((type(a).__module__, type(a).__name__))
            if full_name == "diffusers.configuration_utils.FrozenDict":
                return a
            if full_name == "diffusers.models.attention_processor.AttnProcessor2_0":
                return a
            
            # if (
            #     type(a).__module__.startswith("torch") == False
            #     and type(a).__module__.startswith("diffusers") == False
            # ): 
            #     # print_red(f"found {type(a).__module__} {type(a).__name__} {attribute}")
            #     #"can't be a torch module at this point! But found " + str(type(a))
            #     print_green(f"found {type(a).__module__} {type(a).__name__} {attribute}")
            # else:
            #     print_red (f"found {type(a).__module__} {type(a).__name__} {attribute}")
            #     # import pdb; pdb.set_trace()
            #     # a = replace_obj(a)
            return a

            
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        replacement = replace_class(type(self._1f_proxy_submod))

        if replacement is not None:
            return replacement.__call__(self, *args, **kwargs)
        else:
            raise RuntimeError(
                "can't find oneflow module for: " + str(type(self._1f_proxy_submod))
            )

