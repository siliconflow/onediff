"""
code from https://github.com/comfyanonymous/ComfyUI/blob/4103f7fad5be7e22ed61843166b72b7c41671d75/comfy/ldm/modules/attention.py#L450-L490
"""
import comfy
import oneflow as torch
import oneflow.nn as nn
from typing import Optional, Any

from onediff.infer_compiler.transform import proxy_class


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d


class SpatialTransformer(proxy_class(comfy.ldm.modules.attention.SpatialTransformer)):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
        dtype=None,
        device=None,
        operations=comfy.ops,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels, dtype=dtype, device=device)
        if not use_linear:
            self.proj_in = operations.Conv2d(
                in_channels,
                inner_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=dtype,
                device=device,
            )
        else:
            self.proj_in = operations.Linear(
                in_channels, inner_dim, dtype=dtype, device=device
            )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = operations.Conv2d(
                inner_dim,
                in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=dtype,
                device=device,
            )
        else:
            self.proj_out = operations.Linear(
                in_channels, inner_dim, dtype=dtype, device=device
            )
        self.use_linear = use_linear

    def forward(self, x, context=None, transformer_options={}):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        # NOTE: rearrange in ComfyUI is replaced with reshape and use -1 to enable for
        # dynamic shape inference (multi resolution compilation)
        x = x.flatten(2, 3).permute(0, 2, 1)
        # x = x.reshape(b, c, -1).permute(0, 2, 1)
        # x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            x = block(x, context=context[i], transformer_options=transformer_options)
        if self.use_linear:
            x = self.proj_out(x)
        # NOTE: rearrange in ComfyUI is replaced with permute
        x = x.permute(0, 2, 1).reshape_as(x_in)
        # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class CrossAttention(proxy_class(comfy.ldm.modules.attention.CrossAttention)):
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

    # None of attention methods from the original codes is utilized
    # https://github.com/comfyanonymous/ComfyUI/blob/777f6b15225197898a5f49742682a2be859072d7/comfy/ldm/modules/attention.py#L333-L351
    def forward(self, x, context=None, value=None, mask=None):
        if (
            not exists(context)
            and not exists(value)
            and hasattr(self, "to_qkv")
            and self.to_qkv is not None
        ):
            qkv = self.to_qkv(x)
            out = torch._C.fused_multi_head_attention_inference_v2(
                query=qkv,
                query_head_size=self.dim_head,
                query_layout="BM(H3K)",
                output_layout="BM(HK)",
                scale=self.scale,
                causal=False,
            )
            return self.to_out(out)

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

        return self.to_out(out)
