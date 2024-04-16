"""align attention with ipadapter
https://github.com/cubiq/ComfyUI_InstantID/blob/main/CrossAttentionPatch.py
https://github.com/cubiq/ComfyUI_IPAdapter_plus/blob/main/CrossAttentionPatch.py
"""
import math
import oneflow as torch
import oneflow.nn.functional as F


def tensor_to_size(source, dest_size):
    if isinstance(dest_size, torch.Tensor):
        dest_size = dest_size.shape[0]
    source_size = source.shape[0]

    if source_size < dest_size:
        shape = [dest_size - source_size] + [1] * (source.dim() - 1)
        source = torch.cat((source, source[-1:].repeat(shape)), dim=0)
    elif source_size > dest_size:
        source = source[:dest_size]

    return source


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


class CrossAttentionPatch:
    # forward for patching
    def __init__(
        self,
        ipadapter=None,
        number=0,
        weight=1.0,
        cond=None,
        cond_alt=None,
        uncond=None,
        weight_type="linear",
        mask=None,
        sigma_start=0.0,
        sigma_end=1.0,
        unfold_batch=False,
        embeds_scaling="V only",
    ):
        super().__init__()
        self.weights = [torch.tensor(weight)]
        self.ipadapters = [ipadapter]
        self.conds = [cond]
        self.conds_alt = [cond_alt]
        self.unconds = [uncond]
        self.weight_types = [weight_type]
        self.masks = [mask]
        self.sigma_starts = [sigma_start]
        self.sigma_ends = [sigma_end]
        self.unfold_batch = [unfold_batch]
        self.embeds_scaling = [embeds_scaling]
        self.number = number
        self.layers = (
            11 if "101_to_k_ip" in ipadapter.ip_layers.to_kvs else 16
        )  # TODO: check if this is a valid condition to detect all models

        self.k_key = str(self.number * 2 + 1) + "_to_k_ip"
        self.v_key = str(self.number * 2 + 1) + "_to_v_ip"

        self.cache_map = {}

    def set_new_condition(
        self,
        ipadapter=None,
        number=0,
        weight=1.0,
        cond=None,
        cond_alt=None,
        uncond=None,
        weight_type="linear",
        mask=None,
        sigma_start=0.0,
        sigma_end=1.0,
        unfold_batch=False,
        embeds_scaling="V only",
    ):
        self.weights.append(torch.tensor(weight))
        self.ipadapters.append(ipadapter)
        self.conds.append(cond)
        self.conds_alt.append(cond_alt)
        self.unconds.append(uncond)
        self.weight_types.append(weight_type)
        self.masks.append(mask)
        self.sigma_starts.append(sigma_start)
        self.sigma_ends.append(sigma_end)
        self.unfold_batch.append(unfold_batch)
        self.embeds_scaling.append(embeds_scaling)

    def __call__(self, q, k, v, extra_options):
        dtype = q.dtype
        cond_or_uncond = extra_options["cond_or_uncond"]
        sigma = extra_options["sigmas"] if "sigmas" in extra_options else 999999999.9
        block_type = extra_options["block"][0]
        # block_id = extra_options["block"][1]
        t_idx = extra_options["transformer_index"]

        # extra options for AnimateDiff
        ad_params = extra_options["ad_params"] if "ad_params" in extra_options else None

        b = q.shape[0]
        seq_len = q.shape[1]
        batch_prompt = b // len(cond_or_uncond)
        out = optimized_attention(q, k, v, extra_options["n_heads"])
        _, _, oh, ow = extra_options["original_shape"]

        for (
            weight,
            cond,
            cond_alt,
            uncond,
            ipadapter,
            mask,
            weight_type,
            sigma_start,
            sigma_end,
            unfold_batch,
            embeds_scaling,
        ) in zip(
            self.weights,
            self.conds,
            self.conds_alt,
            self.unconds,
            self.ipadapters,
            self.masks,
            self.weight_types,
            self.sigma_starts,
            self.sigma_ends,
            self.unfold_batch,
            self.embeds_scaling,
        ):
            if sigma <= sigma_start and sigma >= sigma_end:
                if weight_type == "ease in":
                    weight = weight * (0.05 + 0.95 * (1 - t_idx / self.layers))
                elif weight_type == "ease out":
                    weight = weight * (0.05 + 0.95 * (t_idx / self.layers))
                elif weight_type == "ease in-out":
                    weight = weight * (
                        0.05
                        + 0.95
                        * (1 - abs(t_idx - (self.layers / 2)) / (self.layers / 2))
                    )
                elif weight_type == "reverse in-out":
                    weight = weight * (
                        0.05
                        + 0.95 * (abs(t_idx - (self.layers / 2)) / (self.layers / 2))
                    )
                elif weight_type == "weak input" and block_type == "input":
                    weight = weight * 0.2
                elif weight_type == "weak middle" and block_type == "middle":
                    weight = weight * 0.2
                elif weight_type == "weak output" and block_type == "output":
                    weight = weight * 0.2
                elif weight_type == "strong middle" and (
                    block_type == "input" or block_type == "output"
                ):
                    weight = weight * 0.2
                elif isinstance(weight, dict):
                    if t_idx not in weight:
                        continue

                    weight = weight[t_idx]

                    if cond_alt is not None and t_idx in cond_alt:
                        cond = cond_alt[t_idx]
                        del cond_alt

                # if weight == 0:
                #     continue

                if unfold_batch and cond.shape[0] > 1:
                    # Check AnimateDiff context window
                    if ad_params is not None and ad_params["sub_idxs"] is not None:
                        # if image length matches or exceeds full_length get sub_idx images
                        if cond.shape[0] >= ad_params["full_length"]:
                            cond = torch.Tensor(cond[ad_params["sub_idxs"]])
                            uncond = torch.Tensor(uncond[ad_params["sub_idxs"]])
                        # otherwise get sub_idxs images
                        else:
                            cond = tensor_to_size(cond, ad_params["full_length"])
                            uncond = tensor_to_size(uncond, ad_params["full_length"])
                            cond = cond[ad_params["sub_idxs"]]
                            uncond = uncond[ad_params["sub_idxs"]]

                    cond = tensor_to_size(cond, batch_prompt)
                    uncond = tensor_to_size(uncond, batch_prompt)

                    k_cond = ipadapter.ip_layers.to_kvs[self.k_key](cond)
                    k_uncond = ipadapter.ip_layers.to_kvs[self.k_key](uncond)
                    v_cond = ipadapter.ip_layers.to_kvs[self.v_key](cond)
                    v_uncond = ipadapter.ip_layers.to_kvs[self.v_key](uncond)
                else:
                    k_cond = ipadapter.ip_layers.to_kvs[self.k_key](cond).repeat(
                        batch_prompt, 1, 1
                    )
                    k_uncond = ipadapter.ip_layers.to_kvs[self.k_key](uncond).repeat(
                        batch_prompt, 1, 1
                    )
                    v_cond = ipadapter.ip_layers.to_kvs[self.v_key](cond).repeat(
                        batch_prompt, 1, 1
                    )
                    v_uncond = ipadapter.ip_layers.to_kvs[self.v_key](uncond).repeat(
                        batch_prompt, 1, 1
                    )

                ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0)
                ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0)

                if embeds_scaling == "K+mean(V) w/ C penalty":
                    scaling = float(ip_k.shape[2]) / 1280.0
                    weight = weight * scaling
                    ip_k = ip_k * weight
                    ip_v_mean = torch.mean(ip_v, dim=1, keepdim=True)
                    ip_v = (ip_v - ip_v_mean) + ip_v_mean * weight
                    out_ip = optimized_attention(
                        q, ip_k, ip_v, extra_options["n_heads"]
                    )
                    del ip_v_mean
                elif embeds_scaling == "K+V w/ C penalty":
                    scaling = float(ip_k.shape[2]) / 1280.0
                    weight = weight * scaling
                    ip_k = ip_k * weight
                    ip_v = ip_v * weight
                    out_ip = optimized_attention(
                        q, ip_k, ip_v, extra_options["n_heads"]
                    )
                elif embeds_scaling == "K+V":
                    ip_k = ip_k * weight
                    ip_v = ip_v * weight
                    out_ip = optimized_attention(
                        q, ip_k, ip_v, extra_options["n_heads"]
                    )
                else:
                    # ip_v = ip_v * weight
                    out_ip = optimized_attention(
                        q, ip_k, ip_v, extra_options["n_heads"]
                    )
                    out_ip = (
                        out_ip * weight
                    )  # I'm doing this to get the same results as before

                if mask is not None:
                    mask_h = oh / math.sqrt(oh * ow / seq_len)
                    mask_h = int(mask_h) + int((seq_len % int(mask_h)) != 0)
                    mask_w = seq_len // mask_h

                    # check if using AnimateDiff and sliding context window
                    if (
                        mask.shape[0] > 1
                        and ad_params is not None
                        and ad_params["sub_idxs"] is not None
                    ):
                        # if mask length matches or exceeds full_length, get sub_idx masks
                        if mask.shape[0] >= ad_params["full_length"]:
                            mask = torch.Tensor(mask[ad_params["sub_idxs"]])
                            mask = F.interpolate(
                                mask.unsqueeze(1),
                                size=(mask_h, mask_w),
                                mode="bilinear",
                            ).squeeze(1)
                        else:
                            mask = F.interpolate(
                                mask.unsqueeze(1),
                                size=(mask_h, mask_w),
                                mode="bilinear",
                            ).squeeze(1)
                            mask = tensor_to_size(mask, ad_params["full_length"])
                            mask = mask[ad_params["sub_idxs"]]
                    else:
                        mask = F.interpolate(
                            mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear"
                        ).squeeze(1)
                        mask = tensor_to_size(mask, batch_prompt)

                    mask = mask.repeat(len(cond_or_uncond), 1, 1)
                    mask = mask.view(mask.shape[0], -1, 1).repeat(1, 1, out.shape[2])

                    # covers cases where extreme aspect ratios can cause the mask to have a wrong size
                    mask_len = mask_h * mask_w
                    if mask_len < seq_len:
                        pad_len = seq_len - mask_len
                        pad1 = pad_len // 2
                        pad2 = pad_len - pad1
                        mask = F.pad(mask, (0, 0, pad1, pad2), value=0.0)
                    elif mask_len > seq_len:
                        crop_start = (mask_len - seq_len) // 2
                        mask = mask[:, crop_start : crop_start + seq_len, :]

                    out_ip = out_ip * mask

                out = out + out_ip

        return out.to(dtype=dtype)

    def to(self, *args, **kwargs):
        return self

    def set_cache(self, key, value):
        self.cache_map[key] = value

    def retrieve_from_cache(self, key, default=None):
        return self.cache_map.get(key, default)

    def update(self, idx=0, patch_kwargs={}):
        # TODO support
        print(f"Warning: {type(self)} Dynamic modification is not supported.")

        weight = patch_kwargs.pop("weight")
        self.weights[idx].copy_(torch.tensor(weight))

        cond = patch_kwargs.pop("cond")
        self.conds[idx].copy_(cond)

        uncond = patch_kwargs.pop("uncond")
        self.unconds[idx].copy_(uncond)

        # patch_weight_type = patch_kwargs.pop("weight_type")

        # sigma_start = patch_kwargs.pop("sigma_start")
        # self.sigma_start[0] = sigma_start

        # sigma_end = patch_kwargs.pop("sigma_end")
        # self.sigma_end[0] = sigma_end
