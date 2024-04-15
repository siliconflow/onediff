from register_comfy.CrossAttentionPatch import \
    CrossAttentionPatch as CrossAttentionPatch_OF

from onediff.infer_compiler.transform import torch2oflow

from ..utils.booster_utils import is_using_oneflow_backend
from ._config import comfyui_instantid_hijacker, comfyui_instantid_pt

set_model_patch_replace_fn_pt = comfyui_instantid_pt.InstantID._set_model_patch_replace

def get_patches_replace_attn2(diff_model):
    if not hasattr(diff_model, "patches_replace_attn2"):
        diff_model.patches_replace_attn2 = {}
    return diff_model.patches_replace_attn2

def _set_model_patch_replace(org_fn, model, patch_kwargs, key):
    to = model.model_options["transformer_options"]
    diff_model = model.model.diffusion_model
    patches_replace_attn2 = get_patches_replace_attn2(diff_model)

    diff_model.use_cross_attention_patch = True
    patch_kwargs = torch2oflow(patch_kwargs)
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    if key not in to["patches_replace"]["attn2"]:
        if key in patches_replace_attn2:
            patch = patches_replace_attn2[key]
            patch.update(patch_kwargs)
        else:
            patch = CrossAttentionPatch_OF(**patch_kwargs)
            patches_replace_attn2[key] = patch
        to["patches_replace"]["attn2"][key] = patch
    else:
        to["patches_replace"]["attn2"][key].set_new_condition(**patch_kwargs)


def cond_func(org_fn, model, *args,**kwargs):
    return is_using_oneflow_backend(model)


comfyui_instantid_hijacker.register(set_model_patch_replace_fn_pt, _set_model_patch_replace, cond_func)