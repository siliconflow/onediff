from comfy.ldm.modules.attention import attention_pytorch
from register_comfy.CrossAttentionPatch import \
    CrossAttentionPatch as CrossAttentionPatch_PT

from register_comfy.CrossAttentionPatch_v2 import \
    Attn2Replace, ipadapter_attention

from onediff.infer_compiler.transform import torch2oflow
from ..utils.booster_utils import clear_deployable_module_cache_and_unbind
from ..patch_management import PatchType, create_patch_executor

from onediff.infer_compiler.utils.cost_util import cost_time

@cost_time(debug=True, message="set_model_patch_replace_v2")
def set_model_patch_replace_v2(org_fn, model, patch_kwargs, key):
    diff_model = model.model.diffusion_model
    cache_patch_executor = create_patch_executor(PatchType.CachedCrossAttentionPatch)
    unet_extra_options_patch_executor = create_patch_executor(PatchType.UNetExtraInputOptions)
    cache_dict = cache_patch_executor.get_patch(diff_model)
    ui_cache_key = create_patch_executor(PatchType.UiNodeWithIndexPatch).get_patch(model)
    unet_extra_options = unet_extra_options_patch_executor.get_patch(diff_model)

    if "attn2" not in unet_extra_options:
        unet_extra_options["attn2"] = {}

    to = model.model_options["transformer_options"].copy()
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    else:
        to["patches_replace"]["attn2"] = to["patches_replace"]["attn2"].copy()
    
    # patch_kwargs = {
    #     "ipadapter": ipa,
    #     "weight": weight,
    #     "cond": cond,
    #     "cond_alt": cond_alt,
    #     "uncond": uncond,
    #     "weight_type": weight_type,
    #     "mask": attn_mask,
    #     "sigma_start": sigma_start,
    #     "sigma_end": sigma_end,
    #     "unfold_batch": unfold_batch,
    #     "embeds_scaling": embeds_scaling,
    # }
    def split_patch_kwargs(patch_kwargs):
        split1dict = {}
        split2dict = {}
        for k, v in patch_kwargs.items():
            if k in ["cond", "uncond", "mask", "weight"]:
                split1dict[k] = v
            else:
                split2dict[k] = v
        
        return split1dict, split2dict

    new_patch_kwargs, patch_kwargs = split_patch_kwargs(patch_kwargs)
    # update patch_kwargs
    if key in cache_dict:
        try:
            attn2_m = cache_dict[key]
            index = attn2_m.cache_map.get(ui_cache_key, None)
            if index is not None:
                unet_extra_options["attn2"][attn2_m.forward_patch_key][index] = new_patch_kwargs
                return 
        except Exception as e:
            clear_deployable_module_cache_and_unbind(model)


    if key not in to["patches_replace"]["attn2"]:
        if key not in cache_dict:
            attn2_m_pt = Attn2Replace(ipadapter_attention, **patch_kwargs)
            attn2_m_of = torch2oflow(attn2_m_pt, bypass_check=True)
            cache_dict[key] = attn2_m_of
            attn2_m: Attn2Replace = attn2_m_of
            index = len(attn2_m.callback) - 1
            attn2_m.cache_map[ui_cache_key] = index
            unet_extra_options["attn2"][attn2_m.forward_patch_key] = [new_patch_kwargs]
        else:
            attn2_m = cache_dict[key]

        to["patches_replace"]["attn2"][key] = attn2_m
        model.model_options["transformer_options"] = to
    else:
        attn2_m: Attn2Replace = to["patches_replace"]["attn2"][key]
        attn2_m.add(attn2_m.callback[0], **torch2oflow(patch_kwargs))
        unet_extra_options["attn2"][attn2_m.forward_patch_key].append(new_patch_kwargs) # update last patch
        attn2_m.cache_map[ui_cache_key] = len(attn2_m.callback) - 1

        

# def set_model_patch_replace(org_fn, model, patch_kwargs, key):
#     diff_model = model.model.diffusion_model
#     cache_patch_executor = create_patch_executor(PatchType.CachedCrossAttentionPatch)
#     masks_patch_executor = create_patch_executor(PatchType.CrossAttentionForwardMasksPatch)
#     cache_dict = cache_patch_executor.get_patch(diff_model)
#     cache_key = create_patch_executor(PatchType.UiNodeWithIndexPatch).get_patch(model)
#     to = model.model_options["transformer_options"]
    
#     if "patches_replace" not in to:
#         to["patches_replace"] = {}
#     if "attn2" not in to["patches_replace"]:
#         to["patches_replace"]["attn2"] = {}
    
#     masks_dict = masks_patch_executor.get_patch(diff_model)

#     if key in cache_dict:
#         patch: CrossAttentionPatch_PT = cache_dict[key]
#         if patch.retrieve_from_cache(cache_key) is not None:
#             if patch.update(cache_key, torch2oflow(patch_kwargs)):
#                 patch.update_mask(cache_key, masks_dict, patch_kwargs["mask"])
#                 return 
#             else:
#                 clear_deployable_module_cache_and_unbind(model)

#     if key not in to["patches_replace"]["attn2"]:
#         if key not in cache_dict:
#             patch_pt = CrossAttentionPatch_PT(**patch_kwargs)
#             patch_pt.optimized_attention = attention_pytorch
#             patch_of = torch2oflow(patch_pt)
#             patch_of.bind_model(patch_pt)

#             patch = patch_of
#             cache_dict[key] = patch
#             patch.set_cache(cache_key, len(patch.weights) - 1)
#             patch.append_mask(masks_dict, patch_kwargs["mask"])

#         patch: CrossAttentionPatch_PT = cache_dict[key]
#         to["patches_replace"]["attn2"][key] = patch
#     else:
#         patch = to["patches_replace"]["attn2"][key]
#         patch.set_new_condition(**torch2oflow(patch_kwargs))
#         patch.set_cache(cache_key, len(patch.weights) - 1)
#         patch.append_mask(masks_dict, patch_kwargs["mask"])

#         if patch.get_bind_model() is not None:
#             bind_model: CrossAttentionPatch_PT = patch.get_bind_model()
#             bind_model.set_new_condition(**patch_kwargs)
        
    
#     create_patch_executor(PatchType.QuantizedInputPatch).set_patch()