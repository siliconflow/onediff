from functools import partial

from ..booster_utils import is_using_nexfort_backend

from ..hijack_ipadapter_plus.set_model_patch_replace import set_model_patch_replace
from ._config import pulid_comfyui, pulid_comfyui_hijacker

# ComfyUI/custom_nodes/PuLID_ComfyUI/pulid.py
set_model_patch_replace_fn = pulid_comfyui.pulid.set_model_patch_replace
pulid_attention = pulid_comfyui.pulid.pulid_attention


set_model_patch_replace_puild = partial(
    set_model_patch_replace, attention_func=pulid_attention
)


def cond_func(org_fn, model, *args, **kwargs):
    return is_using_nexfort_backend(model)


pulid_comfyui_hijacker.register(
    set_model_patch_replace_fn, set_model_patch_replace_puild, cond_func
)
