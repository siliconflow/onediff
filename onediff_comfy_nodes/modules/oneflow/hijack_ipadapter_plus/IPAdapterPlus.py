"""hijack ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py"""
import os

from register_comfy.CrossAttentionPatch import \
    CrossAttentionPatch as CrossAttentionPatch_OF

from onediff.infer_compiler.deployable_module import DeployableModule
from onediff.infer_compiler.transform import torch2oflow

from ..utils.booster_utils import is_using_oneflow_backend
from ._config import ipadapter_plus_hijacker, ipadapter_plus_pt

set_model_patch_replace_fn_pt = ipadapter_plus_pt.IPAdapterPlus.set_model_patch_replace


def get_patches_replace_attn2(diff_model):
    if not hasattr(diff_model, "patches_replace_attn2"):
        diff_model.patches_replace_attn2 = {}
    return diff_model.patches_replace_attn2


def set_model_patch_replace_fn_of(org_fn, model, patch_kwargs, key):

    patch_kwargs = torch2oflow(patch_kwargs)
    diff_model = model.model.diffusion_model
    diff_model.use_cross_attention_patch = True
    patches_replace_attn2 = get_patches_replace_attn2(diff_model)

    to = model.model_options["transformer_options"]

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


def cond_func(org_fn, model, *args, **kwargs):
    return is_using_oneflow_backend(model)


ipadapter_plus_hijacker.register(
    set_model_patch_replace_fn_pt, set_model_patch_replace_fn_of, cond_func
)
