"""hijack ComfyUI/comfy/utils.py"""
import torch
from comfy.utils import copy_to_param
from ..sd_hijack_utils import Hijacker


def copy_to_param_of(org_fn, obj, attr, value):
    # inplace update tensor instead of replacing it
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])

    if prev.data.dtype == torch.int8 and prev.data.dtype != value.dtype:
        return

    prev.data.copy_(value)


def cond_func(orig_func, *args, **kwargs):
    return True


comfy_utils_hijack = Hijacker()

comfy_utils_hijack.register(
    orig_func=copy_to_param, sub_func=copy_to_param_of, cond_func=cond_func
)
