"""hijack ComfyUI/comfy/utils.py"""
import torch
from comfy.utils import copy_to_param
from onediff.infer_compiler.backends.oneflow.param_utils import (
    update_graph_related_tensor,
)

from ..sd_hijack_utils import Hijacker


@torch.no_grad()
def copy_to_param_of(org_fn, obj, attr, value):
    # inplace update tensor instead of replacing it
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)

    prev = getattr(obj, attrs[-1])

    if prev.data.dtype == torch.int8 and prev.data.dtype != value.dtype:
        return

    prev.data.copy_(value)

    if isinstance(obj, torch.nn.Conv2d):
        update_graph_related_tensor(obj)


def cond_func(orig_func, *args, **kwargs):
    return True


comfy_utils_hijack = Hijacker()

comfy_utils_hijack.register(
    orig_func=copy_to_param, sub_func=copy_to_param_of, cond_func=cond_func
)
