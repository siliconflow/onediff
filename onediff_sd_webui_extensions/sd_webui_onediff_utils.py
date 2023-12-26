from inspect import isfunction

import oneflow as flow
from oneflow import nn


# https://github.com/Stability-AI/generative-models/blob/059d8e9cd9c55aea1ef2ece39abf605efb8b7cc9/sgm/modules/diffusionmodules/util.py#L274
# https://github.com/Stability-AI/stablediffusion/blob/b4bdae9916f628461e1e4edbc62aafedebb9f7ed/ldm/modules/diffusionmodules/util.py#L224
class GroupNorm32Oflow(nn.GroupNorm):
    def forward(self, x):
        # return super().forward(x.float()).type(x.dtype)
        return super().forward(x)


# https://github.com/Stability-AI/generative-models/blob/e5963321482a091a78375f3aeb2c3867562c913f/sgm/modules/diffusionmodules/openaimodel.py#L983-L984
class TimeEmbedModule(nn.Module):
    def __init__( self, time_embed):
        super().__init__()
        self._time_embed_module = time_embed

    def forward(self, t_emb):
        return self._time_embed_module(t_emb.half())


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack_optimizations.py#L142
# https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack_optimizations.py#L221
class CrossAttentionOflow(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        h = self.heads

        q_in = self.to_q(x)
        context = default(context, x)

        # context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
        context_k, context_v = context, context
        k_in = self.to_k(context_k)
        v_in = self.to_v(context_v)
        out = flow._C.fused_multi_head_attention_inference_v2(
            query=q_in,
            query_layout="BM(HK)",
            query_head_size=self.to_q.out_features//self.heads,
            key=k_in,
            key_layout="BM(HK)",
            value=v_in,
            value_layout="BM(HK)",
            output_layout="BM(HK)",
            causal=False,
        )
        return self.to_out(out)
