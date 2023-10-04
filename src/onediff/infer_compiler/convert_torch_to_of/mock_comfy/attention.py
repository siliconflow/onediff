"""
code from https://github.com/comfyanonymous/ComfyUI/blob/4103f7fad5be7e22ed61843166b72b7c41671d75/comfy/ldm/modules/attention.py#L450-L490
"""
import mock_comfy as comfy
import oneflow as torch
import oneflow.nn as nn
from typing import Optional, Any


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d


class CrossAttentionPytorch(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        dtype=None,
        device=None,
        operations=comfy.ops,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = operations.Linear(
            query_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.to_k = operations.Linear(
            context_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.to_v = operations.Linear(
            context_dim, inner_dim, bias=False, dtype=dtype, device=device
        )

        self.to_out = nn.Sequential(
            operations.Linear(inner_dim, query_dim, dtype=dtype, device=device),
            nn.Dropout(dropout),
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, value=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        b, _, _ = q.shape

        head_dim = self.dim_head
        out = torch._C.fused_multi_head_attention_inference_v2(
            query=q,
            query_layout="BM(HK)",
            query_head_size=head_dim,
            key=k,
            key_layout="BM(HK)",
            value=v,
            value_layout="BM(HK)",
            output_layout="BM(HK)",
            causal=False,
        )

        if exists(mask):
            raise NotImplementedError

        if out.ndim != 3:
            print("out.ndim !=3")
            out = out.transpose(1, 2).reshape(b, -1, self.heads * self.dim_head)
        return self.to_out(out)
