from register_comfy.CrossAttentionPatch import (
    CrossAttentionPatch as CrossAttentionPatch_OF,
)

from onediff.infer_compiler.transform import torch2oflow

from ..patch_management import PatchType, create_patch_executor
from ..utils.booster_utils import is_using_oneflow_backend
from ._config import comfyui_instantid_hijacker, comfyui_instantid_pt

set_model_patch_replace_fn_pt = comfyui_instantid_pt.InstantID._set_model_patch_replace


def _set_model_patch_replace(org_fn, model, patch_kwargs, key):

    patch_kwargs = torch2oflow(patch_kwargs)
    diff_model = model.model.diffusion_model
    cache_patch_executor = create_patch_executor(PatchType.C_C_Patch)
    cache_dict = cache_patch_executor.get_patch(diff_model)
    cau_patch_executor = create_patch_executor(PatchType.CrossAttentionUpdatePatch)

    cache_key = cau_patch_executor.get_patch(model)
    to = model.model_options["transformer_options"]

    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}

    if key in cache_dict:
        patch: CrossAttentionPatch_OF = cache_dict[key]
        patch.update(cache_key, patch_kwargs)

    if key not in to["patches_replace"]["attn2"]:
        if key not in cache_dict:
            patch = CrossAttentionPatch_OF(**patch_kwargs)
            cache_dict[key] = patch
            patch.set_cache(cache_key, len(patch.weights) - 1)

        patch: CrossAttentionPatch_OF = cache_dict[key]

        to["patches_replace"]["attn2"][key] = patch
    else:
        patch = to["patches_replace"]["attn2"][key]
        patch.set_new_condition(**patch_kwargs)
        patch.set_cache(cache_key, len(patch.weights) - 1)


def cond_func(org_fn, model, *args, **kwargs):
    return is_using_oneflow_backend(model)


comfyui_instantid_hijacker.register(
    set_model_patch_replace_fn_pt, _set_model_patch_replace, cond_func
)
