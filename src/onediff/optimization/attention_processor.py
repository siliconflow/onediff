import os
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F


class FusedSelfAttnProcessor:
    def __init__(self, attn):
        assert hasattr(attn, "to_qkv") and attn.to_qkv is not None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        assert (
            encoder_hidden_states is None
        ), "encoder_hidden_states must be None for FusedSelfAttnProcessor"

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        qkv = attn.to_qkv(hidden_states)

        inner_dim = qkv.shape[-1] // 3
        head_dim = inner_dim // attn.heads

        if False:
            qkv = qkv.view(-1, attn.heads, head_dim * 3)

            query = (
                qkv[:, :, 0:head_dim].reshape(batch_size, -1, inner_dim).contiguous()
            )
            key = (
                qkv[:, :, head_dim : head_dim * 2]
                .reshape(batch_size, -1, inner_dim)
                .contiguous()
            )
            value = (
                qkv[:, :, head_dim * 2 : head_dim * 3]
                .reshape(batch_size, -1, inner_dim)
                .contiguous()
            )

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = flow.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
        else:
            from ..infer_compiler.utils import (
                parse_boolean_from_env,
                set_boolean_env_var,
            )

            if attn.upcast_attention and parse_boolean_from_env(
                "ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_ACCUMULATION", True
            ):
                set_boolean_env_var(
                    "ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_ACCUMULATION", False
                )
            hidden_states = flow._C.fused_multi_head_attention_inference_v2(
                query=qkv,
                query_head_size=head_dim,
                query_layout="BM(H3K)",
                output_layout="BM(HK)",
                scale=attn.scale,
                causal=False,
            )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


try:
    from onediff.infer_compiler.transform import register

    def convert_fused_self_attn_processor(
        mod: FusedSelfAttnProcessor, verbose=True
    ) -> FusedSelfAttnProcessor:
        return mod

    register(torch2oflow_funcs=convert_fused_self_attn_processor)
except:
    print("Skip onediff.infer_compiler.transform.register")
