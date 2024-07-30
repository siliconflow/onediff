# ComfyUI/comfy/hijack_model_management.py
import oneflow as flow  # usort: skip
from comfy.model_management import soft_empty_cache

from ..sd_hijack_utils import Hijacker


def hijack_soft_empty_cache(original_func, *args, **kwargs):
    original_func(*args, **kwargs)
    if flow.cuda.is_available():
        flow.cuda.empty_cache()


# Hijacker
model_management_hijacker = Hijacker()

model_management_hijacker.register(
    soft_empty_cache, hijack_soft_empty_cache, lambda *args, **kwags: True
)
