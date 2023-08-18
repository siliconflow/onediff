import os
import torch
import diffusers
import oneflow
import oneflow as flow
from torch.fx.experimental.proxy_tensor import make_fx
from torch.func import functionalize
import importlib
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from .attention_1f import BasicTransformerBlock, FeedForward, GEGLU
from .attention_processor_1f import Attention
from .lora_1f import LoRACompatibleLinear


_is_diffusers_quant_available = False
try:
    import diffusers_quant
    _is_diffusers_quant_available = True
except:
    pass


def replace_class(cls):
    if cls.__module__.startswith("torch"):
        mod_name = cls.__module__.replace("torch", "oneflow")
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls.__name__)
    elif cls == diffusers.models.attention.BasicTransformerBlock:
        return BasicTransformerBlock
    elif cls == diffusers.models.attention_processor.Attention:
        return Attention
    elif cls == diffusers.models.attention.FeedForward:
        return FeedForward
    elif cls == diffusers.models.attention.GEGLU:
        return GEGLU
    elif cls == diffusers.models.lora.LoRACompatibleLinear:
        return LoRACompatibleLinear

    if _is_diffusers_quant_available:
        if cls == diffusers_quant.FakeQuantModule:
            return diffusers_quant.OneFlowFakeQuantModule
        if cls == diffusers_quant.StaticQuantModule:
            return diffusers_quant.OneFlowStaticQuantModule
        if cls == diffusers_quant.DynamicQuantModule:
            return diffusers_quant.OneFlowDynamicQuantModule


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
        return oneflow.nn.functional.conv2d
    if func == torch._C._nn.linear:
        return oneflow.nn.functional.linear
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

    def __getattribute__(self, attribute):
        if attribute.startswith("_1f_proxy"):
            return object.__getattribute__(self, attribute)
        elif attribute in ["forward", "_conv_forward"]:
            replacement = replace_class(type(self._1f_proxy_submod))
            return lambda *args, **kwargs: getattr(replacement, attribute)(
                self, *args, **kwargs
            )
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
            elif isinstance(a, torch.nn.Module):
                if attribute not in self._1f_proxy_children:
                    a = ProxySubmodule(a)
                    self._1f_proxy_children[attribute] = a
                else:
                    a = self._1f_proxy_children[attribute]
            assert (
                type(a).__module__.startswith("torch") == False
                and type(a).__module__.startswith("diffusers") == False
            ), "can't be a torch module at this point! But found " + str(type(a))
            return a

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        replacement = replace_class(type(self._1f_proxy_submod))
        if replacement is not None:
            return replacement.__call__(self, *args, **kwargs)
        else:
            raise RuntimeError(
                "can't find oneflow module for: " + str(type(self._1f_proxy_submod))
            )


class OneFlowInterpreter(torch.fx.Interpreter):
    from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate

    def call_function(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
        if target == torch.sigmoid:
            return torch.neg(*args, **kwargs)
        args, kwargs = map_args(args, kwargs)
        target = replace_func(target)
        return super().call_function(target, args, kwargs)

    def call_method(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
        args, kwargs = map_args(args, kwargs)
        return super().call_method(target, args, kwargs)

    def call_module(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        submod = self.fetch_attr(target)
        submod = ProxySubmodule(submod)
        return submod(*args, **kwargs)


def torchbackend(gm, example_inputs):
    def wrapped_forward(*args, **kwargs):
        args = [flow.utils.tensor.from_torch(a) for a in args]
        output = OneFlowInterpreter(gm, garbage_collect_values=False).run(
            *args, **kwargs
        )
        if isinstance(output, tuple):
            return tuple(flow.utils.tensor.to_torch(i) for i in output)
        return flow.utils.tensor.to_torch(output)

    return wrapped_forward
