import importlib
import os
import torch
import oneflow as flow

__of_mds = {}
__convert_list = [
    "diffusers.models.unet_2d_condition.UNet2DConditionModel",
    "diffusers.models.embeddings.TimestepEmbedding",
    "diffusers.models.embeddings.Timesteps",
    "diffusers.models.resnet.ResnetBlock2D",
    "diffusers.models.resnet.Downsample2D",
    "diffusers.models.resnet.Upsample2D",
    "diffusers.models.transformer_2d.Transformer2DModel",
    "diffusers.models.unet_2d_blocks.CrossAttnDownBlock2D",
    "diffusers.models.unet_2d_blocks.DownBlock2D",
    "diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D",
    "diffusers.models.unet_2d_blocks.UpBlock2D",
    "diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D",
    "diffusers.models.unet_2d_blocks.UNetMidBlock2DCrossAttn",
]

with flow.mock_torch.enable(lazy=False):
    for md_name in __convert_list:
        p, m = md_name.rsplit('.', 1)
        md = importlib.import_module(p)
        __of_mds[md_name] = getattr(md, m)

import diffusers
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from .attention_1f import BasicTransformerBlock, FeedForward, GEGLU
from .attention_processor_1f import Attention
from .lora_1f import LoRACompatibleLinear, LoRACompatibleConv

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
    elif cls == diffusers.models.lora.LoRACompatibleConv:
        return LoRACompatibleConv
    
    full_cls_name = str(cls.__module__) + '.' + str(cls.__name__)
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
            # assert (
            #     type(a).__module__.startswith("torch") == False
            #     and type(a).__module__.startswith("diffusers") == False
            # ), "can't be a torch module at this point! But found " + str(type(a))
            return a

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        replacement = replace_class(type(self._1f_proxy_submod))
        if replacement is not None:
            return replacement.__call__(self, *args, **kwargs)
        else:
            raise RuntimeError(
                "can't find oneflow module for: " + str(type(self._1f_proxy_submod))
            )
