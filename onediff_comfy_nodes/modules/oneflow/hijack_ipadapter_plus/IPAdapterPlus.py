"""hijack ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py
commit 2ff4fc482029d408cfd5fa05522ca822b2c2e33c (HEAD -> main, origin/main, origin/HEAD)
Author: cubiq <matteo@elf.io>
Date:   Sun Jun 23 18:22:48 2024 +0200

    enable finetuning of precise style transfer with mad scientist node
"""

import torch
import os
import math
import folder_paths

import comfy.model_management as model_management
from node_helpers import conditioning_set_values
from comfy.clip_vision import load as load_clip_vision
from comfy.sd import load_lora_for_models
import comfy.utils

import torch.nn as nn
from PIL import Image
try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

import comfy.model_management as model_management
from node_helpers import conditioning_set_values
from comfy.clip_vision import load as load_clip_vision
from comfy.sd import load_lora_for_models
import comfy.utils

from ..utils.booster_utils import is_using_oneflow_backend
from ._config import ipadapter_plus_hijacker, ipadapter_plus_pt
from .set_model_patch_replace import set_model_patch_replace_v2

set_model_patch_replace_fn_pt = ipadapter_plus_pt.IPAdapterPlus.set_model_patch_replace
tensor_to_image = ipadapter_plus_pt.utils.tensor_to_image
image_to_tensor = ipadapter_plus_pt.utils.image_to_tensor
encode_image_masked = ipadapter_plus_pt.utils.encode_image_masked
tensor_to_size = ipadapter_plus_pt.utils.tensor_to_size
IPAdapter = ipadapter_plus_pt.IPAdapterPlus.IPAdapter


def set_model_patch_replace(model, patch_kwargs, key):
    set_model_patch_replace_v2(org_fn=None,model=model, patch_kwargs=patch_kwargs, key=key)

id_func = id
def ipadapter_execute(org_fn, model,
                      ipadapter,
                      clipvision,
                      insightface=None,
                      image=None,
                      image_composition=None,
                      image_negative=None,
                      weight=1.0,
                      weight_composition=1.0,
                      weight_faceidv2=None,
                      weight_type="linear",
                      combine_embeds="concat",
                      start_at=0.0,
                      end_at=1.0,
                      attn_mask=None,
                      pos_embed=None,
                      neg_embed=None,
                      unfold_batch=False,
                      embeds_scaling='V only',
                      layer_weights=None,
                      encode_batch_size=0,):
    device = model_management.get_torch_device()
    dtype = model_management.unet_dtype()
    if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        dtype = torch.float16 if model_management.should_use_fp16() else torch.float32

    is_full = "proj.3.weight" in ipadapter["image_proj"]
    is_portrait = "proj.2.weight" in ipadapter["image_proj"] and not "proj.3.weight" in ipadapter["image_proj"] and not "0.to_q_lora.down.weight" in ipadapter["ip_adapter"]
    is_portrait_unnorm = "portraitunnorm" in ipadapter
    is_faceid = is_portrait or "0.to_q_lora.down.weight" in ipadapter["ip_adapter"] or is_portrait_unnorm
    is_plus = (is_full or "latents" in ipadapter["image_proj"] or "perceiver_resampler.proj_in.weight" in ipadapter["image_proj"]) and not is_portrait_unnorm
    is_faceidv2 = "faceidplusv2" in ipadapter
    output_cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
    is_sdxl = output_cross_attention_dim == 2048

    if is_faceid and not insightface:
        raise Exception("insightface model is required for FaceID models")

    if is_faceidv2:
        weight_faceidv2 = weight_faceidv2 if weight_faceidv2 is not None else weight*2

    cross_attention_dim = 1280 if (is_plus and is_sdxl and not is_faceid) or is_portrait_unnorm else output_cross_attention_dim
    clip_extra_context_tokens = 16 if (is_plus and not is_faceid) or is_portrait or is_portrait_unnorm else 4

    if image is not None and image.shape[1] != image.shape[2]:
        print("\033[33mINFO: the IPAdapter reference image is not a square, CLIPImageProcessor will resize and crop it at the center. If the main focus of the picture is not in the middle the result might not be what you are expecting.\033[0m")

    if isinstance(weight, list):
        weight = torch.tensor(weight).unsqueeze(-1).unsqueeze(-1).to(device, dtype=dtype) if unfold_batch else weight[0]

    # special weight types
    if layer_weights is not None and layer_weights != '':
        weight = { int(k): float(v)*weight for k, v in [x.split(":") for x in layer_weights.split(",")] }
        weight_type = weight_type if weight_type == "style transfer precise" else "linear"
    elif weight_type == "style transfer":
        weight = { 6:weight } if is_sdxl else { 0:weight, 1:weight, 2:weight, 3:weight, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "composition":
        weight = { 3:weight } if is_sdxl else { 4:weight*0.25, 5:weight }
    elif weight_type == "strong style transfer":
        if is_sdxl:
            weight = { 0:weight, 1:weight, 2:weight, 4:weight, 5:weight, 6:weight, 7:weight, 8:weight, 9:weight, 10:weight }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 6:weight, 7:weight, 8:weight, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "style and composition":
        if is_sdxl:
            weight = { 3:weight_composition, 6:weight }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 4:weight_composition*0.25, 5:weight_composition, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "strong style and composition":
        if is_sdxl:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight_composition, 4:weight, 5:weight, 6:weight, 7:weight, 8:weight, 9:weight, 10:weight }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 4:weight_composition, 5:weight_composition, 6:weight, 7:weight, 8:weight, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "style transfer precise":
        if is_sdxl:
            weight = { 3:weight, 6:weight }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 4:weight*0.25, 5:weight, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }

    img_comp_cond_embeds = None
    face_cond_embeds = None
    if is_faceid:
        if insightface is None:
            raise Exception("Insightface model is required for FaceID models")

        from insightface.utils import face_align

        insightface.det_model.input_size = (640,640) # reset the detection size
        image_iface = tensor_to_image(image)
        face_cond_embeds = []
        image = []

        for i in range(image_iface.shape[0]):
            for size in [(size, size) for size in range(640, 256, -64)]:
                insightface.det_model.input_size = size # TODO: hacky but seems to be working
                face = insightface.get(image_iface[i])
                if face:
                    if not is_portrait_unnorm:
                        face_cond_embeds.append(torch.from_numpy(face[0].normed_embedding).unsqueeze(0))
                    else:
                        face_cond_embeds.append(torch.from_numpy(face[0].embedding).unsqueeze(0))
                    image.append(image_to_tensor(face_align.norm_crop(image_iface[i], landmark=face[0].kps, image_size=256 if is_sdxl else 224)))

                    if 640 not in size:
                        print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
                    break
            else:
                raise Exception('InsightFace: No face detected.')
        face_cond_embeds = torch.stack(face_cond_embeds).to(device, dtype=dtype)
        image = torch.stack(image)
        del image_iface, face

    if image is not None:
        img_cond_embeds = encode_image_masked(clipvision, image, batch_size=encode_batch_size)
        if image_composition is not None:
            img_comp_cond_embeds = encode_image_masked(clipvision, image_composition, batch_size=encode_batch_size)

        if is_plus:
            img_cond_embeds = img_cond_embeds.penultimate_hidden_states
            image_negative = image_negative if image_negative is not None else torch.zeros([1, 224, 224, 3])
            img_uncond_embeds = encode_image_masked(clipvision, image_negative, batch_size=encode_batch_size).penultimate_hidden_states
            if image_composition is not None:
                img_comp_cond_embeds = img_comp_cond_embeds.penultimate_hidden_states
        else:
            img_cond_embeds = img_cond_embeds.image_embeds if not is_faceid else face_cond_embeds
            if image_negative is not None and not is_faceid:
                img_uncond_embeds = encode_image_masked(clipvision, image_negative, batch_size=encode_batch_size).image_embeds
            else:
                img_uncond_embeds = torch.zeros_like(img_cond_embeds)
            if image_composition is not None:
                img_comp_cond_embeds = img_comp_cond_embeds.image_embeds
        del image_negative, image_composition

        image = None if not is_faceid else image # if it's face_id we need the cropped face for later
    elif pos_embed is not None:
        img_cond_embeds = pos_embed

        if neg_embed is not None:
            img_uncond_embeds = neg_embed
        else:
            if is_plus:
                img_uncond_embeds = encode_image_masked(clipvision, torch.zeros([1, 224, 224, 3])).penultimate_hidden_states
            else:
                img_uncond_embeds = torch.zeros_like(img_cond_embeds)
        del pos_embed, neg_embed
    else:
        raise Exception("Images or Embeds are required")

    # ensure that cond and uncond have the same batch size
    img_uncond_embeds = tensor_to_size(img_uncond_embeds, img_cond_embeds.shape[0])

    img_cond_embeds = img_cond_embeds.to(device, dtype=dtype)
    img_uncond_embeds = img_uncond_embeds.to(device, dtype=dtype)
    if img_comp_cond_embeds is not None:
        img_comp_cond_embeds = img_comp_cond_embeds.to(device, dtype=dtype)

    # combine the embeddings if needed
    if combine_embeds != "concat" and img_cond_embeds.shape[0] > 1 and not unfold_batch:
        if combine_embeds == "add":
            img_cond_embeds = torch.sum(img_cond_embeds, dim=0).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.sum(face_cond_embeds, dim=0).unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = torch.sum(img_comp_cond_embeds, dim=0).unsqueeze(0)
        elif combine_embeds == "subtract":
            img_cond_embeds = img_cond_embeds[0] - torch.mean(img_cond_embeds[1:], dim=0)
            img_cond_embeds = img_cond_embeds.unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = face_cond_embeds[0] - torch.mean(face_cond_embeds[1:], dim=0)
                face_cond_embeds = face_cond_embeds.unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = img_comp_cond_embeds[0] - torch.mean(img_comp_cond_embeds[1:], dim=0)
                img_comp_cond_embeds = img_comp_cond_embeds.unsqueeze(0)
        elif combine_embeds == "average":
            img_cond_embeds = torch.mean(img_cond_embeds, dim=0).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.mean(face_cond_embeds, dim=0).unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = torch.mean(img_comp_cond_embeds, dim=0).unsqueeze(0)
        elif combine_embeds == "norm average":
            img_cond_embeds = torch.mean(img_cond_embeds / torch.norm(img_cond_embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.mean(face_cond_embeds / torch.norm(face_cond_embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = torch.mean(img_comp_cond_embeds / torch.norm(img_comp_cond_embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
        img_uncond_embeds = img_uncond_embeds[0].unsqueeze(0) # TODO: better strategy for uncond could be to average them

    if attn_mask is not None:
        attn_mask = attn_mask.to(device, dtype=dtype)


    # import time 
    # to = time.time()
    diff_model = model.model.diffusion_model
    # Create a cache key based on the attributes
    cache_key = (
            id_func(ipadapter),
            cross_attention_dim,
            output_cross_attention_dim,
            img_cond_embeds.shape[-1],
            clip_extra_context_tokens,
            is_sdxl,
            is_plus,
            is_full,
            is_faceid,
            is_portrait_unnorm
        )
    if hasattr(diff_model, '_ipa_info') and cache_key in diff_model._ipa_info:
        print(f'Use cached {id_func(ipadapter)=}')
        ipa = diff_model._ipa_info[cache_key]
    else:
        ipa = IPAdapter(
            ipadapter,
            cross_attention_dim=cross_attention_dim,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=img_cond_embeds.shape[-1],
            clip_extra_context_tokens=clip_extra_context_tokens,
            is_sdxl=is_sdxl,
            is_plus=is_plus,
            is_full=is_full,
            is_faceid=is_faceid,
            is_portrait_unnorm=is_portrait_unnorm,
        ).to(device, dtype=dtype)
        # TODO 
        if not hasattr(diff_model, "_ipa_info"):
            diff_model._ipa_info =  {cache_key: ipa}
        else:
            diff_model._ipa_info[cache_key] = ipa

    # print(f'274 {time.time() - to} sec')

    if is_faceid and is_plus:
        cond = ipa.get_image_embeds_faceid_plus(face_cond_embeds, img_cond_embeds, weight_faceidv2, is_faceidv2, encode_batch_size)
        # TODO: check if noise helps with the uncond face embeds
        uncond = ipa.get_image_embeds_faceid_plus(torch.zeros_like(face_cond_embeds), img_uncond_embeds, weight_faceidv2, is_faceidv2, encode_batch_size)
    else:
        cond, uncond = ipa.get_image_embeds(img_cond_embeds, img_uncond_embeds, encode_batch_size)
        if img_comp_cond_embeds is not None:
            cond_comp = ipa.get_image_embeds(img_comp_cond_embeds, img_uncond_embeds, encode_batch_size)[0]

    cond = cond.to(device, dtype=dtype)
    uncond = uncond.to(device, dtype=dtype)

    cond_alt = None
    if img_comp_cond_embeds is not None:
        cond_alt = { 3: cond_comp.to(device, dtype=dtype) }

    del img_cond_embeds, img_uncond_embeds, img_comp_cond_embeds, face_cond_embeds

    sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
    sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

    patch_kwargs = {
        "ipadapter": ipa,
        "weight": weight,
        "cond": cond,
        "cond_alt": cond_alt,
        "uncond": uncond,
        "weight_type": weight_type,
        "mask": attn_mask,
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
        "unfold_batch": unfold_batch,
        "embeds_scaling": embeds_scaling,
    }

    number = 0
    if not is_sdxl:
        for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
            patch_kwargs["module_key"] = str(number*2+1)
            set_model_patch_replace(model, patch_kwargs, ("input", id))
            number += 1
        for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
            patch_kwargs["module_key"] = str(number*2+1)
            set_model_patch_replace(model, patch_kwargs, ("output", id))
            number += 1
        patch_kwargs["module_key"] = str(number*2+1)
        set_model_patch_replace(model, patch_kwargs, ("middle", 0))
    else:
        for id in [4,5,7,8]: # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(model, patch_kwargs, ("input", id, index))
                number += 1
        for id in range(6): # id of output_blocks that have cross attention
            block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(model, patch_kwargs, ("output", id, index))
                number += 1
        for index in range(10):
            patch_kwargs["module_key"] = str(number*2+1)
            set_model_patch_replace(model, patch_kwargs, ("middle", 0, index))
            number += 1

    return (model, image)



def cond_func(org_fn, model, *args, **kwargs):
    return is_using_oneflow_backend(model)
    

ipadapter_plus_hijacker.register(
    ipadapter_plus_pt.IPAdapterPlus.ipadapter_execute, ipadapter_execute, cond_func)