import os
from pathlib import Path
from ._config import ipadapter_plus_pt, ipadapter_plus_of,ipadapter_plus_hijacker
import oneflow as torch 
import oneflow.nn.functional as F
import math
from onediff.infer_compiler.with_oneflow_compile import DeployableModule
# ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py
set_model_patch_replace_fn_pt = ipadapter_plus_pt.IPAdapterPlus.set_model_patch_replace

def attention_pytorch(q, k, v, heads, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    # q, k, v = map(
    #     lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
    #     (q, k, v),
    # )

    # out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    # out = (
    #     out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    # )
    assert mask is None
    out = torch._C.fused_multi_head_attention_inference_v2(
        query=q,
        query_layout="BM(HK)",
        query_head_size=dim_head,
        key=k,
        key_layout="BM(HK)",
        value=v,
        value_layout="BM(HK)",
        output_layout="BM(HK)",
        causal=False,
    )

    return out

optimized_attention = attention_pytorch
class CrossAttentionPatch_OF(torch.nn.Module):
    # forward for patching
    def __init__(self, weight, ipadapter, number, cond, uncond, weight_type, mask=None, sigma_start=0.0, sigma_end=1.0, unfold_batch=False):
        super().__init__()

        self.weights = [weight]
        self.ipadapters = [ipadapter]
        self.conds = [cond]
        self.unconds = [uncond]
        self.number = number
        self.weight_type = [weight_type]
        self.masks = [mask]
        self.sigma_start = [sigma_start]
        self.sigma_end = [sigma_end]
        self.unfold_batch = [unfold_batch]

        self.k_key = str(self.number*2+1) + "_to_k_ip"
        self.v_key = str(self.number*2+1) + "_to_v_ip"
    
    def set_new_condition(self, weight, ipadapter, number, cond, uncond, weight_type, mask=None, sigma_start=0.0, sigma_end=1.0, unfold_batch=False):
        self.weights.append(weight)
        self.ipadapters.append(ipadapter)
        self.conds.append(cond)
        self.unconds.append(uncond)
        self.masks.append(mask)
        self.weight_type.append(weight_type)
        self.sigma_start.append(sigma_start)
        self.sigma_end.append(sigma_end)
        self.unfold_batch.append(unfold_batch)

    def forward(self, n, context_attn2, value_attn2, extra_options):
        org_dtype = n.dtype
        cond_or_uncond = extra_options["cond_or_uncond"]
        sigma = extra_options["sigmas"][0].item() if 'sigmas' in extra_options else 999999999.9

        # extra options for AnimateDiff
        ad_params = extra_options['ad_params'] if "ad_params" in extra_options else None

        q = n
        k = context_attn2
        v = value_attn2
        b = q.shape[0]
        qs = q.shape[1]
        batch_prompt = b // len(cond_or_uncond)
        # out = optimized_attention(q, k, v, extra_options["n_heads"])
        out = attention_pytorch(q, k, v, extra_options["n_heads"])
        _, _, lh, lw = extra_options["original_shape"]
        
        for weight, cond, uncond, ipadapter, mask, weight_type, sigma_start, sigma_end, unfold_batch in zip(self.weights, self.conds, self.unconds, self.ipadapters, self.masks, self.weight_type, self.sigma_start, self.sigma_end, self.unfold_batch):
            if sigma > sigma_start or sigma < sigma_end:
                continue

            if unfold_batch and cond.shape[0] > 1:
                # Check AnimateDiff context window
                if ad_params is not None and ad_params["sub_idxs"] is not None:
                    # if images length matches or exceeds full_length get sub_idx images
                    if cond.shape[0] >= ad_params["full_length"]:
                        cond = torch.Tensor(cond[ad_params["sub_idxs"]])
                        uncond = torch.Tensor(uncond[ad_params["sub_idxs"]])
                    # otherwise, need to do more to get proper sub_idxs masks
                    else:
                        # check if images length matches full_length - if not, make it match
                        if cond.shape[0] < ad_params["full_length"]:
                            cond = torch.cat((cond, cond[-1:].repeat((ad_params["full_length"]-cond.shape[0], 1, 1))), dim=0)
                            uncond = torch.cat((uncond, uncond[-1:].repeat((ad_params["full_length"]-uncond.shape[0], 1, 1))), dim=0)
                        # if we have too many remove the excess (should not happen, but just in case)
                        if cond.shape[0] > ad_params["full_length"]:
                            cond = cond[:ad_params["full_length"]]
                            uncond = uncond[:ad_params["full_length"]]
                        cond = cond[ad_params["sub_idxs"]]
                        uncond = uncond[ad_params["sub_idxs"]]

                # if we don't have enough reference images repeat the last one until we reach the right size
                if cond.shape[0] < batch_prompt:
                    cond = torch.cat((cond, cond[-1:].repeat((batch_prompt-cond.shape[0], 1, 1))), dim=0)
                    uncond = torch.cat((uncond, uncond[-1:].repeat((batch_prompt-uncond.shape[0], 1, 1))), dim=0)
                # if we have too many remove the exceeding
                elif cond.shape[0] > batch_prompt:
                    cond = cond[:batch_prompt]
                    uncond = uncond[:batch_prompt]

                k_cond = ipadapter.ip_layers.to_kvs[self.k_key](cond)
                k_uncond = ipadapter.ip_layers.to_kvs[self.k_key](uncond)
                v_cond = ipadapter.ip_layers.to_kvs[self.v_key](cond)
                v_uncond = ipadapter.ip_layers.to_kvs[self.v_key](uncond)
            else:
                k_cond = ipadapter.ip_layers.to_kvs[self.k_key](cond).repeat(batch_prompt, 1, 1)
                k_uncond = ipadapter.ip_layers.to_kvs[self.k_key](uncond).repeat(batch_prompt, 1, 1)
                v_cond = ipadapter.ip_layers.to_kvs[self.v_key](cond).repeat(batch_prompt, 1, 1)
                v_uncond = ipadapter.ip_layers.to_kvs[self.v_key](uncond).repeat(batch_prompt, 1, 1)

            if weight_type.startswith("linear"):
                ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0) * weight
                ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0) * weight
            else:
                ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0)
                ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0)

                if weight_type.startswith("channel"):
                    # code by Lvmin Zhang at Stanford University as also seen on Fooocus IPAdapter implementation
                    ip_v_mean = torch.mean(ip_v, dim=1, keepdim=True)
                    ip_v_offset = ip_v - ip_v_mean
                    _, _, C = ip_k.shape
                    channel_penalty = float(C) / 1280.0
                    W = weight * channel_penalty
                    ip_k = ip_k * W
                    ip_v = ip_v_offset + ip_v_mean * W

            out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])           
            if weight_type.startswith("original"):
                out_ip = out_ip * weight

            if mask is not None:
                # TODO: needs checking
                mask_h = lh / math.sqrt(lh * lw / qs)
                mask_h = int(mask_h) + int((qs % int(mask_h)) != 0)
                mask_w = qs // mask_h

                # check if using AnimateDiff and sliding context window
                if (mask.shape[0] > 1 and ad_params is not None and ad_params["sub_idxs"] is not None):
                    # if mask length matches or exceeds full_length, just get sub_idx masks, resize, and continue
                    if mask.shape[0] >= ad_params["full_length"]:
                        mask_downsample = torch.Tensor(mask[ad_params["sub_idxs"]])
                        mask_downsample = F.interpolate(mask_downsample.unsqueeze(1), size=(mask_h, mask_w), mode="bicubic").squeeze(1)
                    # otherwise, need to do more to get proper sub_idxs masks
                    else:
                        # resize to needed attention size (to save on memory)
                        mask_downsample = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bicubic").squeeze(1)
                        # check if mask length matches full_length - if not, make it match
                        if mask_downsample.shape[0] < ad_params["full_length"]:
                            mask_downsample = torch.cat((mask_downsample, mask_downsample[-1:].repeat((ad_params["full_length"]-mask_downsample.shape[0], 1, 1))), dim=0)
                        # if we have too many remove the excess (should not happen, but just in case)
                        if mask_downsample.shape[0] > ad_params["full_length"]:
                            mask_downsample = mask_downsample[:ad_params["full_length"]]
                        # now, select sub_idxs masks
                        mask_downsample = mask_downsample[ad_params["sub_idxs"]]
                # otherwise, perform usual mask interpolation
                else:
                    mask_downsample = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bicubic").squeeze(1)

                # if we don't have enough masks repeat the last one until we reach the right size
                if mask_downsample.shape[0] < batch_prompt:
                    mask_downsample = torch.cat((mask_downsample, mask_downsample[-1:, :, :].repeat((batch_prompt-mask_downsample.shape[0], 1, 1))), dim=0)
                # if we have too many remove the exceeding
                elif mask_downsample.shape[0] > batch_prompt:
                    mask_downsample = mask_downsample[:batch_prompt, :, :]
                
                # repeat the masks
                mask_downsample = mask_downsample.repeat(len(cond_or_uncond), 1, 1)
                mask_downsample = mask_downsample.view(mask_downsample.shape[0], -1, 1).repeat(1, 1, out.shape[2])

                out_ip = out_ip * mask_downsample

            out = out + out_ip

        return out.to(dtype=org_dtype)
    
    def to(self, *args, **kwargs):
        # print("Warning: CrossAttentionPatch_OF.to() is called, but it is not implemented yet.")
        return self

def set_model_patch_replace_fn_of(org_fn, model, patch_kwargs, key):
    from onediff.infer_compiler.transform import torch2oflow
    from onediff.infer_compiler import oneflow_compile
    patch_kwargs = torch2oflow(patch_kwargs)
    model.model.diffusion_model = oneflow_compile(model.model.diffusion_model, use_graph=True, dynamic=False, options={})

    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    if key not in to["patches_replace"]["attn2"]:
        patch = CrossAttentionPatch_OF(**patch_kwargs)
        to["patches_replace"]["attn2"][key] = patch
    else:
        to["patches_replace"]["attn2"][key].set_new_condition(**patch_kwargs)

def cond_func(org_fn, model, *args, **kwargs):
    return isinstance(model.model.diffusion_model, DeployableModule)
ipadapter_plus_hijacker.register(set_model_patch_replace_fn_pt,set_model_patch_replace_fn_of,cond_func)