from typing import List

import ldm.modules.attention
import onediff_shared
import sgm.modules.attention
import torch
from ldm.modules.diffusionmodules.util import timestep_embedding
from modules import shared
from modules.hypernetworks import hypernetwork
from modules.sd_hijack_optimizations import SdOptimization
from modules.sd_hijack_utils import CondFunc

from onediff.utils.import_utils import is_nexfort_available
from onediff_utils import singleton_decorator


@singleton_decorator
def init_nexfort_backend():
    CondFunc(
        "ldm.modules.diffusionmodules.openaimodel.UNetModel.forward",
        onediff_nexfort_unet_ldm_forward,
        lambda orig_func, *args, **kwargs: onediff_shared.onediff_enabled,
    )

    CondFunc(
        "sgm.modules.diffusionmodules.openaimodel.UNetModel.forward",
        onediff_nexfort_unet_sgm_forward,
        lambda orig_func, *args, **kwargs: onediff_shared.onediff_enabled,
    )

    def hijack_groupnorm32_forward(orig_func, self, x):
        return super(type(self), self).forward(x)
        # return self.forward(x)

    CondFunc(
        "ldm.modules.diffusionmodules.util.GroupNorm32.forward",
        hijack_groupnorm32_forward,
        lambda orig_func, *args, **kwargs: onediff_shared.onediff_enabled,
    )
    CondFunc(
        "sgm.modules.diffusionmodules.util.GroupNorm32.forward",
        hijack_groupnorm32_forward,
        lambda orig_func, *args, **kwargs: onediff_shared.onediff_enabled,
    )


@torch.autocast("cuda", enabled=False)
def onediff_nexfort_unet_sgm_forward(
    orig_func, self, x, timesteps=None, context=None, y=None, **kwargs
):
    assert (y is not None) == (
        self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"
    hs = []
    t_emb = timestep_embedding(timesteps, self.model_channels).half()
    emb = self.time_embed(t_emb)
    x = x.half()
    context = context.half() if context is not None else context
    y = y.half() if y is not None else y
    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)
    h = x
    for module in self.input_blocks:
        h = module(h, emb, context)
        hs.append(h)
    h = self.middle_block(h, emb, context)
    for module in self.output_blocks:
        h = torch.cat([h, hs.pop()], dim=1)
        h = module(h, emb, context)
    h = h.type(x.dtype)
    return self.out(h)


@torch.autocast("cuda", enabled=False)
def onediff_nexfort_unet_ldm_forward(
    orig_func, self, x, timesteps=None, context=None, y=None, **kwargs
):
    assert (y is not None) == (
        self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"
    hs = []
    t_emb = timestep_embedding(timesteps, self.model_channels).half()
    emb = self.time_embed(t_emb)
    x = x.half()
    context = context.half()
    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)
    h = x
    for module in self.input_blocks:
        h = module(h, emb, context)
        hs.append(h)
    h = self.middle_block(h, emb, context)
    for module in self.output_blocks:
        h = torch.cat([h, hs.pop()], dim=1)
        h = module(h, emb, context)
    if self.predict_codebook_ids:
        return self.id_predictor(h)
    else:
        return self.out(h)


def scaled_dot_product_attention_forward(self, x, context=None, mask=None, **kwargs):
    batch_size, sequence_length, inner_dim = x.shape

    if mask is not None:
        mask = self.prepare_attention_mask(mask, sequence_length, batch_size)
        mask = mask.view(batch_size, self.heads, -1, mask.shape[-1])

    h = self.heads
    q_in = self.to_q(x)

    context = x if context is None else context

    context_k, context_v = hypernetwork.apply_hypernetworks(
        shared.loaded_hypernetworks, context
    )
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    head_dim = inner_dim // h
    q = q_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
    k = k_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
    v = v_in.view(batch_size, -1, h, head_dim).transpose(1, 2)

    del q_in, k_in, v_in

    dtype = q.dtype

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    hidden_states = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, h * head_dim)
    hidden_states = hidden_states.to(dtype)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)
    return hidden_states


class SdOptimizationNexfort(SdOptimization):
    name = "nexfort"
    cmd_opt = "nexfort"
    priority = 10

    def is_available(self):
        return is_nexfort_available()

    def apply(self):
        ldm.modules.attention.CrossAttention.forward = (
            scaled_dot_product_attention_forward
        )
        sgm.modules.attention.CrossAttention.forward = (
            scaled_dot_product_attention_forward
        )


def add_nexfort_optimizer(res: List):
    res.append(SdOptimizationNexfort())
