"""
code from https://github.com/comfyanonymous/ComfyUI/blob/4103f7fad5be7e22ed61843166b72b7c41671d75/comfy/ldm/modules/attention.py#L450-L490
"""
from abc import abstractmethod
from typing import Any, Optional

import comfy
import oneflow as torch
import oneflow.nn as nn
from einops import rearrange, repeat
from onediff.infer_compiler.backends.oneflow.transform import proxy_class, transform_mgr

onediff_comfy = transform_mgr.transform_package("comfy")

ops = onediff_comfy.ops.disable_weight_init
timestep_embedding = onediff_comfy.ldm.modules.diffusionmodules.util.timestep_embedding


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


class SpatialVideoTransformer(
    proxy_class(comfy.ldm.modules.attention.SpatialVideoTransformer)
):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        timesteps=None,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        time_context_dim=None,
        ff_in=False,
        checkpoint=False,
        time_depth=1,
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        max_time_embed_period: int = 10000,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            use_checkpoint=checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period

        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim

        self.time_stack = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=time_context_dim,
                    # timesteps=timesteps,
                    checkpoint=checkpoint,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for _ in range(self.depth)
            ]
        )

        assert len(self.time_stack) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = self.in_channels * 4
        self.time_pos_embed = nn.Sequential(
            operations.Linear(
                self.in_channels, time_embed_dim, dtype=dtype, device=device
            ),
            nn.SiLU(),
            operations.Linear(
                time_embed_dim, self.in_channels, dtype=dtype, device=device
            ),
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor, merge_strategy=merge_strategy
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        transformer_options={},
    ) -> torch.Tensor:
        _, _, h, w = x.shape
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context
        if self.use_spatial_context:
            assert (
                context.ndim == 3
            ), f"n dims of spatial context should be 3 but are {context.ndim}"

            if time_context is None:
                time_context = context
            time_context_first_timestep = time_context[::timesteps]
            # time_context = repeat(
            #     time_context_first_timestep, "b ... -> (b n) ...", n=h * w
            # )
            # Rewrite for onediff SVD dynamic shape
            time_context = torch._C.broadcast_dim_like(
                time_context_first_timestep[None, :],
                x.flatten(2, 3),
                dim=0,
                like_dim=2,
            ).flatten(0, 1)

        elif time_context is not None and not self.use_spatial_context:
            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            time_context = torch._C.broadcast_dim_like(
                time_context_first_timestep[None, :],
                x.flatten(2, 3),
                dim=0,
                like_dim=2,
            )
            if time_context.ndim == 2:
                time_context = rearrange(time_context, "b c -> b 1 c")
                # time_context = time_context.unsqueeze(1)

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        # x = rearrange(x, "b c h w -> b (h w) c")
        # Rewrite for onediff SVD dynamic shape
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        if self.use_linear:
            x = self.proj_in(x)

        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        # num_frames = rearrange(num_frames, "b t -> (b t)")
        # Rewrite for onediff SVD dynamic shape
        num_frames = num_frames.flatten()
        t_emb = timestep_embedding(
            num_frames,
            self.in_channels,
            repeat_only=False,
            max_period=self.max_time_embed_period,
        ).to(x.dtype)
        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        for it_, (block, mix_block) in enumerate(
            zip(self.transformer_blocks, self.time_stack)
        ):
            transformer_options["block_index"] = it_
            x = block(
                x,
                context=spatial_context,
                transformer_options=transformer_options,
            )

            x_mix = x
            x_mix = x_mix + emb

            B, S, C = x_mix.shape
            # x_mix = rearrange(x_mix, "(b t) s c -> (b s) t c", t=timesteps)
            # Rewrite for onediff SVD dynamic shape
            b = B // timesteps
            x_mix = x_mix.unflatten(0, shape=(b, -1)).permute(0, 2, 1, 3).flatten(0, 1)
            x_mix = mix_block(x_mix, context=time_context)  # TODO: transformer_options
            # x_mix = rearrange(
            #     x_mix, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
            # )
            # Rewrite for onediff SVD dynamic shape
            x_mix = x_mix.unflatten(0, shape=(b, -1)).permute(0, 2, 1, 3).flatten(0, 1)

            x = self.time_mixer(
                x_spatial=x, x_temporal=x_mix, image_only_indicator=image_only_indicator
            )

        if self.use_linear:
            x = self.proj_out(x)
        # x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        # Rewrite for onediff SVD dynamic shape
        x = x.reshape_as(x_in.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out


def attention_pytorch_oneflow(q, k, v, heads, mask=None, attn_precision=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    head_dim = dim_head
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
    return out
