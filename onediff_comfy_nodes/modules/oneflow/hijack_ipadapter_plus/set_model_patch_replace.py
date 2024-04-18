from comfy.ldm.modules.attention import attention_pytorch
from register_comfy.CrossAttentionPatch import \
    CrossAttentionPatch as CrossAttentionPatch_PT

from onediff.infer_compiler.transform import torch2oflow

from ..patch_management import PatchType, create_patch_executor


def set_model_patch_replace(org_fn, model, patch_kwargs, key):
    diff_model = model.model.diffusion_model
    cache_patch_executor = create_patch_executor(PatchType.CachedCrossAttentionPatch)
    cache_dict = cache_patch_executor.get_patch(diff_model)
    cache_key = create_patch_executor(PatchType.UiNodeWithIndexPatch).get_patch(model)
    to = model.model_options["transformer_options"]
    
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}

    if key in cache_dict:
        patch: CrossAttentionPatch_PT = cache_dict[key]
        if patch.retrieve_from_cache(cache_key) is not None:
            # TODO fix 
            patch.update(cache_key, **torch2oflow(patch_kwargs))
            # cache_patch_executor.set_patch(diff_model, {})
            # diff_model._clear_old_graph()
            # del cache_dict
            # cache_dict = cache_patch_executor.get_patch(diff_model)
            return 
                



    if key not in to["patches_replace"]["attn2"]:
        if key not in cache_dict:
            patch_pt = CrossAttentionPatch_PT(**patch_kwargs)
            patch_pt.optimized_attention = attention_pytorch
            patch_of = torch2oflow(patch_pt)
            patch_of.bind_model(patch_pt)

            patch = patch_of
            cache_dict[key] = patch
            patch.set_cache(cache_key, len(patch.weights) - 1)
        patch: CrossAttentionPatch_PT = cache_dict[key]
        to["patches_replace"]["attn2"][key] = patch
    else:
        patch = to["patches_replace"]["attn2"][key]
        patch.set_new_condition(**torch2oflow(patch_kwargs))
        patch.set_cache(cache_key, len(patch.weights) - 1)
    
    create_patch_executor(PatchType.QuantizedInputPatch).set_patch()