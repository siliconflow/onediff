"""hijack ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py"""
from ..utils.booster_utils import is_using_oneflow_backend
from ._config import ipadapter_plus_hijacker, ipadapter_plus_pt
from .set_model_patch_replace import set_model_patch_replace_v2
set_model_patch_replace_fn_pt = ipadapter_plus_pt.IPAdapterPlus.set_model_patch_replace


def cond_func(org_fn, model, *args, **kwargs):
    return is_using_oneflow_backend(model)


ipadapter_plus_hijacker.register(
    set_model_patch_replace_fn_pt, set_model_patch_replace_v2, cond_func
)
