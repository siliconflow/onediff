"""hijack ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py"""
from ..booster_utils import is_using_nexfort_backend
from ._config import ipadapter_plus, ipadapter_plus_hijacker
from .set_model_patch_replace import set_model_patch_replace

set_model_patch_replace_fn = ipadapter_plus.IPAdapterPlus.set_model_patch_replace


def cond_func(org_fn, model, *args, **kwargs):
    return is_using_nexfort_backend(model)


ipadapter_plus_hijacker.register(
    set_model_patch_replace_fn, set_model_patch_replace, cond_func
)
