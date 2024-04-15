from ._config import comfyui_instantid_pt, comfyui_instantid_of, comfyui_instantid_hijacker
# from onediff.infer_compiler.oneflow import OneflowDeployableModule
from onediff.infer_compiler.transform import torch2oflow

from .CrossAttentionPatch import CrossAttentionPatch
from ..utils.booster_utils import is_using_oneflow_backend

set_model_patch_replace_fn_pt = comfyui_instantid_pt.InstantID._set_model_patch_replace

def _set_model_patch_replace(org_fn, model, patch_kwargs, key):
    to = model.model_options["transformer_options"]
    diff_model = model.model.diffusion_model
    diff_model.use_cross_attention_patch = True
    patch_kwargs = torch2oflow(patch_kwargs)
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = CrossAttentionPatch(**patch_kwargs)
    else:
        to["patches_replace"]["attn2"][key].set_new_condition(**patch_kwargs)


def cond_func(org_fn, model, *args,**kwargs):
    return is_using_oneflow_backend(model)
    # return isinstance(model.model.diffusion_model, OneflowDeployableModule)


comfyui_instantid_hijacker.register(set_model_patch_replace_fn_pt, _set_model_patch_replace, cond_func)