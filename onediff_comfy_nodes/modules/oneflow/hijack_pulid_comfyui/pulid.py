from register_comfy.CrossAttentionPatch import pulid_attention

from ..hijack_ipadapter_plus.set_model_patch_replace import apply_patch
from ..utils.booster_utils import is_using_oneflow_backend
from ._config import pulid_comfyui_hijacker, pulid_comfyui_pt

set_model_patch_replace_pt = pulid_comfyui_pt.pulid.set_model_patch_replace


def set_model_patch_replace_of(org_fn, model, patch_kwargs, key):
    apply_patch(
        org_fn,
        model=model,
        patch_kwargs=patch_kwargs,
        key=key,
        attention_func=pulid_attention,
    )


def cond_func(org_fn, model, *args, **kwargs):
    return is_using_oneflow_backend(model)


pulid_comfyui_hijacker.register(
    set_model_patch_replace_pt, set_model_patch_replace_of, cond_func
)
