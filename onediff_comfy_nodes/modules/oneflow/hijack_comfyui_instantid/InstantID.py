from ..utils.booster_utils import is_using_oneflow_backend
from ._config import comfyui_instantid_hijacker, comfyui_instantid_pt
from ..hijack_ipadapter_plus.set_model_patch_replace import set_model_patch_replace_v2

set_model_patch_replace_fn_pt = comfyui_instantid_pt.InstantID._set_model_patch_replace


def cond_func(org_fn, model, *args, **kwargs):
    return is_using_oneflow_backend(model)


comfyui_instantid_hijacker.register(
    set_model_patch_replace_fn_pt, set_model_patch_replace_v2, cond_func
)
