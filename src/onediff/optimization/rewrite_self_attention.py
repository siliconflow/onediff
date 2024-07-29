import os

import torch
import torch.nn as nn

from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
)

from .attention_processor import FusedSelfAttnProcessor

_IS_ONEDIFF_QUANT_AVAILABLE = 0
try:
    import onediff_quant
    from onediff_quant import DynamicQuantLinearModule, StaticQuantLinearModule

    _IS_ONEDIFF_QUANT_AVAILABLE = 1
except ImportError as e:
    pass


def rewrite_self_attention(model):
    if "GeForce RTX" in torch.cuda.get_device_name():
        return

    def is_attention_can_be_fused(attn):
        dim_head = attn.to_q.out_features // attn.heads
        if dim_head not in (40, 64):
            return False
        if attn.to_k is None or attn.to_v is None:
            return False
        if (
            attn.to_q.bias is not None
            or attn.to_k.bias is not None
            or attn.to_v.bias is not None
        ):
            return False
        if not isinstance(attn.processor, (AttnProcessor, AttnProcessor2_0)):
            return False
        if (
            attn.to_q.in_features != attn.to_k.in_features
            or attn.to_q.in_features != attn.to_v.in_features
        ):
            return False
        if not (
            attn.to_q.weight.dtype == attn.to_k.weight.dtype
            and attn.to_q.weight.dtype == attn.to_v.weight.dtype
        ):
            return False
        return True

    def rewrite_attention(attn):
        dim_head = attn.to_q.out_features // attn.heads
        has_bias = attn.to_q.bias is not None
        attn.to_qkv = nn.Linear(
            attn.to_q.in_features, attn.to_q.out_features * 3, bias=has_bias
        )
        attn.to_qkv.requires_grad_(False)

        qkv_weight = torch.cat(
            [
                attn.to_q.weight.permute(1, 0).reshape(-1, attn.heads, dim_head),
                attn.to_k.weight.permute(1, 0).reshape(-1, attn.heads, dim_head),
                attn.to_v.weight.permute(1, 0).reshape(-1, attn.heads, dim_head),
            ],
            dim=2,
        )
        qkv_weight = (
            qkv_weight.reshape(-1, attn.to_q.out_features * 3)
            .permute(1, 0)
            .contiguous()
        )
        attn.to_qkv.weight.data = qkv_weight

        if has_bias:
            qkv_bias = (
                torch.cat(
                    [
                        attn.to_q.bias.reshape(attn.heads, dim_head),
                        attn.to_k.bias.reshape(attn.heads, dim_head),
                        attn.to_v.bias.reshape(attn.heads, dim_head),
                    ],
                    dim=1,
                )
                .reshape(attn.to_q.out_features * 3)
                .contiguous()
            )
            attn.to_qkv.bias.data = qkv_bias

        if _IS_ONEDIFF_QUANT_AVAILABLE and isinstance(
            attn.to_q, (DynamicQuantLinearModule, StaticQuantLinearModule)
        ):
            cls = type(attn.to_q)
            weight_scale = (
                torch.cat(
                    [
                        torch.Tensor(attn.to_q.calibrate[2]).reshape(
                            attn.heads, dim_head
                        ),
                        torch.Tensor(attn.to_k.calibrate[2]).reshape(
                            attn.heads, dim_head
                        ),
                        torch.Tensor(attn.to_v.calibrate[2]).reshape(
                            attn.heads, dim_head
                        ),
                    ],
                    dim=1,
                )
                .reshape(attn.to_q.out_features * 3)
                .contiguous()
            )
            calibrate = [attn.to_q.calibrate[0], attn.to_q.calibrate[1], weight_scale]
            attn.to_qkv = cls(attn.to_qkv, attn.to_q.nbits, calibrate, attn.to_q.name)

        del attn.to_q
        del attn.to_k
        del attn.to_v
        attn.processor = FusedSelfAttnProcessor(attn)

    for m in model.modules():
        if isinstance(m, Attention) and is_attention_can_be_fused(m):
            rewrite_attention(m)
