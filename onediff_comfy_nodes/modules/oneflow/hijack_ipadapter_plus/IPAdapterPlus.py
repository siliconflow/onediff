"""hijack ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py"""
import os
from pathlib import Path

from onediff.infer_compiler import DeployableModule
from onediff.infer_compiler.transform import torch2oflow

from ._config import ipadapter_plus_hijacker, ipadapter_plus_of, ipadapter_plus_pt
from .CrossAttentionPatch import CrossAttentionPatch as CrossAttentionPatch_OF

set_model_patch_replace_fn_pt = ipadapter_plus_pt.IPAdapterPlus.set_model_patch_replace


def get_patches_replace_attn2(diff_model):
    if not hasattr(diff_model._deployable_module_model, "patches_replace_attn2"):
        diff_model._deployable_module_model.patches_replace_attn2 = {}
    return diff_model._deployable_module_model.patches_replace_attn2


def set_model_patch_replace_fn_of(org_fn, model, patch_kwargs, key):

    patch_kwargs = torch2oflow(patch_kwargs)
    diff_model = model.model.diffusion_model
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
    return isinstance(model.model.diffusion_model, DeployableModule)


ipadapter_plus_hijacker.register(
    set_model_patch_replace_fn_pt, set_model_patch_replace_fn_of, cond_func
)
