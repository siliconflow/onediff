from comfy.ldm.modules.attention import attention_pytorch
from register_comfy.CrossAttentionPatch import \
    CrossAttentionPatch as CrossAttentionPatch_PT

from onediff.infer_compiler.transform import torch2oflow
from ..utils.booster_utils import clear_deployable_module_cache_and_unbind
from ..patch_management import PatchType, create_patch_executor


def set_model_patch_replace(org_fn, model, patch_kwargs, key):
    diff_model = model.model.diffusion_model
    cache_patch_executor = create_patch_executor(PatchType.CachedCrossAttentionPatch)
    masks_patch_executor = create_patch_executor(PatchType.CrossAttentionForwardMasksPatch)
    cache_dict = cache_patch_executor.get_patch(diff_model)
    cache_key = create_patch_executor(PatchType.UiNodeWithIndexPatch).get_patch(model)
    to = model.model_options["transformer_options"]
    
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    
    masks_dict = masks_patch_executor.get_patch(diff_model)

    if key in cache_dict:
        patch: CrossAttentionPatch_PT = cache_dict[key]
        if patch.retrieve_from_cache(cache_key) is not None:
            if patch.update(cache_key, torch2oflow(patch_kwargs)):
                patch.update_mask(cache_key, masks_dict, patch_kwargs["mask"])
                return 
            else:
                clear_deployable_module_cache_and_unbind(model)

    if key not in to["patches_replace"]["attn2"]:
        if key not in cache_dict:
            patch_pt = CrossAttentionPatch_PT(**patch_kwargs)
            patch_pt.optimized_attention = attention_pytorch
            patch_of = torch2oflow(patch_pt)
            patch_of.bind_model(patch_pt)

            patch = patch_of
            cache_dict[key] = patch
            patch.set_cache(cache_key, len(patch.weights) - 1)
            patch.append_mask(masks_dict, patch_kwargs["mask"])

        patch: CrossAttentionPatch_PT = cache_dict[key]
        to["patches_replace"]["attn2"][key] = patch
    else:
        patch = to["patches_replace"]["attn2"][key]
        patch.set_new_condition(**torch2oflow(patch_kwargs))
        patch.set_cache(cache_key, len(patch.weights) - 1)
        patch.append_mask(masks_dict, patch_kwargs["mask"])

        if patch.get_bind_model() is not None:
            bind_model: CrossAttentionPatch_PT = patch.get_bind_model()
            bind_model.set_new_condition(**patch_kwargs)
        
    
    create_patch_executor(PatchType.QuantizedInputPatch).set_patch()