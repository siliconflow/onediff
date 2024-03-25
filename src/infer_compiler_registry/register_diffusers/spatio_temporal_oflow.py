# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import oneflow as torch
import oneflow.nn.functional as F
from oneflow import nn

import functools
from typing import Any, Optional, Tuple, Union, Dict
from collections import OrderedDict

from .attention_processor_oflow import AttentionProcessor

from packaging import version
import importlib.metadata

diffusers_version = version.parse(importlib.metadata.version("diffusers"))

diffusers_0240_v = version.parse("0.24.0")

if diffusers_version >= diffusers_0240_v:

    from onediff.infer_compiler.transform import transform_mgr

    transformed_diffusers = transform_mgr.transform_package("diffusers")

    diffusers_0260_v = version.parse("0.26.0")

    if diffusers_version >= diffusers_0260_v:
        DiffusersUNetSpatioTemporalConditionModel = (
            transformed_diffusers.models.unets.unet_spatio_temporal_condition.UNetSpatioTemporalConditionModel
        )
        DiffusersTransformerSpatioTemporalModel = (
            transformed_diffusers.models.transformers.transformer_temporal.TransformerSpatioTemporalModel
        )

    else:
        DiffusersUNetSpatioTemporalConditionModel = (
            transformed_diffusers.models.unet_spatio_temporal_condition.UNetSpatioTemporalConditionModel
        )
        DiffusersTransformerSpatioTemporalModel = (
            transformed_diffusers.models.transformer_temporal.TransformerSpatioTemporalModel
        )

    if diffusers_version >= version.parse("0.25.00"):
        DiffusersTemporalDecoder = (
            transformed_diffusers.models.autoencoders.autoencoder_kl_temporal_decoder.TemporalDecoder
        )
    else:
        DiffusersTemporalDecoder = (
            transformed_diffusers.models.autoencoder_kl_temporal_decoder.TemporalDecoder
        )

    DiffusersSpatioTemporalResBlock = (
        transformed_diffusers.models.resnet.SpatioTemporalResBlock
    )
    DiffusersTemporalBasicTransformerBlock = (
        transformed_diffusers.models.attention.TemporalBasicTransformerBlock
    )

    class TemporalDecoder(DiffusersTemporalDecoder):
        def forward(
            self,
            sample: torch.FloatTensor,
            image_only_indicator: torch.FloatTensor,
            num_frames: int = 1,
        ) -> torch.FloatTensor:
            r"""The forward method of the `Decoder` class."""

            sample = self.conv_in(sample)

            upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    # middle
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.mid_block),
                        sample,
                        image_only_indicator,
                        use_reentrant=False,
                    )
                    sample = sample.to(upscale_dtype)

                    # up
                    for up_block in self.up_blocks:
                        sample = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(up_block),
                            sample,
                            image_only_indicator,
                            use_reentrant=False,
                        )
                else:
                    # middle
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.mid_block),
                        sample,
                        image_only_indicator,
                    )
                    sample = sample.to(upscale_dtype)

                    # up
                    for up_block in self.up_blocks:
                        sample = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(up_block),
                            sample,
                            image_only_indicator,
                        )
            else:
                # middle
                sample = self.mid_block(
                    sample, image_only_indicator=image_only_indicator
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = up_block(sample, image_only_indicator=image_only_indicator)

            # post-process
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
            sample = self.conv_out(sample)

            batch_frames, channels, height, width = sample.shape
            batch_size = batch_frames // num_frames
            # sample = sample[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
            # Dynamic shape for VAE divide chunks
            sample = sample.unflatten(0, shape=(batch_size, -1)).permute(0, 2, 1, 3, 4)
            sample = self.time_conv_out(sample)

            # sample = sample.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
            # Dynamic shape for VAE divide chunks
            sample = sample.permute(0, 2, 1, 3, 4).flatten(0, 1)

            return sample

    # VideoResBlock
    class SpatioTemporalResBlock(DiffusersSpatioTemporalResBlock):
        def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: Optional[torch.FloatTensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None,
        ):
            num_frames = image_only_indicator.shape[-1]
            hidden_states = self.spatial_res_block(hidden_states, temb)

            batch_frames, channels, height, width = hidden_states.shape
            batch_size = batch_frames // num_frames

            # hidden_states_mix = (
            #     hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
            # )
            # hidden_states = (
            #     hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
            # )
            #
            # Dynamic shape for VAE divide chunks
            hidden_states_mix = hidden_states.unflatten(
                0, shape=(batch_size, -1)
            ).permute(0, 2, 1, 3, 4)
            hidden_states = hidden_states.unflatten(0, shape=(batch_size, -1)).permute(
                0, 2, 1, 3, 4
            )

            if temb is not None:
                # temb = temb.reshape(batch_size, num_frames, -1)
                temb = temb.unflatten(0, shape=(batch_size, -1))

            hidden_states = self.temporal_res_block(hidden_states, temb)
            hidden_states = self.time_mixer(
                x_spatial=hidden_states_mix,
                x_temporal=hidden_states,
                image_only_indicator=image_only_indicator,
            )

            # hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
            # Dynamic shape for VAE divide chunks
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
            return hidden_states

    class TransformerSpatioTemporalModel(DiffusersTransformerSpatioTemporalModel):
        def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None,
            return_dict: bool = True,
        ):
            """
            Args:
                hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                    Input hidden_states.
                num_frames (`int`):
                    The number of frames to be processed per batch. This is used to reshape the hidden states.
                encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                    Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                    self-attention.
                image_only_indicator (`torch.LongTensor` of shape `(batch size, num_frames)`, *optional*):
                    A tensor indicating whether the input contains only images. 1 indicates that the input contains only
                    images, 0 indicates that the input contains video frames.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~models.transformer_temporal.TransformerTemporalModelOutput`] instead of a plain
                    tuple.

            Returns:
                [`~models.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                    If `return_dict` is True, an [`~models.transformer_temporal.TransformerTemporalModelOutput`] is
                    returned, otherwise a `tuple` where the first element is the sample tensor.
            """
            # 1. Input
            batch_frames, _, height, width = hidden_states.shape
            num_frames = image_only_indicator.shape[-1]
            batch_size = batch_frames // num_frames

            time_context = encoder_hidden_states
            # time_context_first_timestep = time_context[None, :].reshape(
            #     batch_size, num_frames, -1, time_context.shape[-1]
            # )[:, 0]
            # Rewrite for onediff SVD dynamic shape
            time_context_first_timestep = time_context.unflatten(
                0, shape=(batch_size, -1)
            )[:, 0]
            # time_context = time_context_first_timestep[None, :].broadcast_to(
            #     height * width, batch_size, 1, time_context.shape[-1]
            # )
            # Rewrite for onediff SVD dynamic shape
            time_context = torch._C.broadcast_dim_like(
                time_context_first_timestep[None, :],
                hidden_states.flatten(2, 3),
                dim=0,
                like_dim=2,
            )
            # time_context = time_context.reshape(height * width * batch_size, 1, time_context.shape[-1])
            # Rewrite for onediff SVD dynamic shape
            time_context = time_context.flatten(0, 1)

            residual = hidden_states

            hidden_states = self.norm(hidden_states)
            inner_dim = hidden_states.shape[1]
            # hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)
            # Rewrite for onediff SVD dynamic shape
            hidden_states = hidden_states.permute(0, 2, 3, 1).flatten(1, 2)

            hidden_states = self.proj_in(hidden_states)

            num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
            num_frames_emb = num_frames_emb.repeat(batch_size, 1)
            num_frames_emb = num_frames_emb.reshape(-1)
            t_emb = self.time_proj(num_frames_emb)

            # `Timesteps` does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=hidden_states.dtype)

            emb = self.time_pos_embed(t_emb)
            emb = emb[:, None, :]

            # 2. Blocks
            for block, temporal_block in zip(
                self.transformer_blocks, self.temporal_transformer_blocks
            ):
                if self.training and self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        block,
                        hidden_states,
                        None,
                        encoder_hidden_states,
                        None,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = block(
                        hidden_states, encoder_hidden_states=encoder_hidden_states,
                    )

                hidden_states_mix = hidden_states
                hidden_states_mix = hidden_states_mix + emb

                hidden_states_mix = temporal_block(
                    hidden_states_mix,
                    num_frames=num_frames,
                    encoder_hidden_states=time_context,
                )
                hidden_states = self.time_mixer(
                    x_spatial=hidden_states,
                    x_temporal=hidden_states_mix,
                    image_only_indicator=image_only_indicator,
                )

            # 3. Output
            hidden_states = self.proj_out(hidden_states)
            # hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            # Rewrite for onediff SVD dynamic shape
            hidden_states = (
                hidden_states.reshape_as(residual.permute(0, 2, 3, 1))
                .permute(0, 3, 1, 2)
                .contiguous()
            )

            output = hidden_states + residual

            if not return_dict:
                return (output,)

            return TransformerTemporalModelOutput(sample=output)

    class TemporalBasicTransformerBlock(DiffusersTemporalBasicTransformerBlock):
        def forward(
            self,
            hidden_states: torch.FloatTensor,
            num_frames: int,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
        ) -> torch.FloatTensor:
            # Notice that normalization is always applied before the real computation in the following blocks.
            # 0. Self-Attention
            batch_size = hidden_states.shape[0]

            batch_frames, seq_length, channels = hidden_states.shape
            batch_size = batch_frames // num_frames

            # hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, seq_length, channels)
            # Rewrite for onediff SVD dynamic shape
            hidden_states = hidden_states.unflatten(0, shape=(batch_size, -1))
            hidden_states = hidden_states.permute(0, 2, 1, 3)
            # hidden_states = hidden_states.reshape(batch_size * seq_length, num_frames, channels)
            # Rewrite for onediff SVD dynamic shape
            hidden_states = hidden_states.flatten(0, 1)

            residual = hidden_states
            hidden_states = self.norm_in(hidden_states)

            if self._chunk_size is not None:
                hidden_states = _chunked_feed_forward(
                    self.ff_in, hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                hidden_states = self.ff_in(hidden_states)

            if self.is_res:
                hidden_states = hidden_states + residual

            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
            hidden_states = attn_output + hidden_states

            # 3. Cross-Attention
            if self.attn2 is not None:
                norm_hidden_states = self.norm2(hidden_states)
                attn_output = self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states
                )
                hidden_states = attn_output + hidden_states

            # 4. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self._chunk_size is not None:
                ff_output = _chunked_feed_forward(
                    self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                ff_output = self.ff(norm_hidden_states)

            if self.is_res:
                hidden_states = ff_output + hidden_states
            else:
                hidden_states = ff_output

            # hidden_states = hidden_states[None, :].reshape(batch_size, seq_length, num_frames, channels)
            # Rewrite for onediff SVD dynamic shape
            hidden_states = hidden_states.unflatten(0, shape=(batch_size, -1))
            hidden_states = hidden_states.permute(0, 2, 1, 3)
            # hidden_states = hidden_states.reshape(batch_size * num_frames, seq_length, channels)
            # Rewrite for onediff SVD dynamic shape
            hidden_states = hidden_states.flatten(0, 1)

            return hidden_states

    class UNetSpatioTemporalConditionModel(DiffusersUNetSpatioTemporalConditionModel):
        def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            added_time_ids: torch.Tensor,
            return_dict: bool = True,
        ) -> Union[Tuple]:
            r"""
            The [`UNetSpatioTemporalConditionModel`] forward method.

            Args:
                sample (`torch.FloatTensor`):
                    The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
                timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
                encoder_hidden_states (`torch.FloatTensor`):
                    The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
                added_time_ids: (`torch.FloatTensor`):
                    The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                    embeddings and added to the time embeddings.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead of a plain
                    tuple.
            Returns:
                [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                    If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is returned, otherwise
                    a `tuple` is returned where the first element is the sample tensor.
            """
            # 1. time
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            batch_size, num_frames = sample.shape[:2]
            timesteps = timesteps.expand(batch_size)

            t_emb = self.time_proj(timesteps)

            # `Timesteps` does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=sample.dtype)

            emb = self.time_embedding(t_emb)

            time_embeds = self.add_time_proj(added_time_ids.flatten())
            time_embeds = time_embeds.reshape((batch_size, -1))
            time_embeds = time_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(time_embeds)
            emb = emb + aug_emb

            # Flatten the batch and frames dimensions
            # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
            sample = sample.flatten(0, 1)
            # Repeat the embeddings num_video_frames times
            # emb: [batch, channels] -> [batch * frames, channels]
            emb = emb.repeat_interleave(num_frames, dim=0)
            # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                num_frames, dim=0
            )

            # 2. pre-process
            sample = self.conv_in(sample)

            image_only_indicator = torch.zeros(
                batch_size, num_frames, dtype=sample.dtype, device=sample.device
            )

            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:
                if (
                    hasattr(downsample_block, "has_cross_attention")
                    and downsample_block.has_cross_attention
                ):
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        image_only_indicator=image_only_indicator,
                    )
                else:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        image_only_indicator=image_only_indicator,
                    )

                down_block_res_samples += res_samples

            # 4. mid
            sample = self.mid_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
            )

            # 5. up
            for i, upsample_block in enumerate(self.up_blocks):
                res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                down_block_res_samples = down_block_res_samples[
                    : -len(upsample_block.resnets)
                ]

                if (
                    hasattr(upsample_block, "has_cross_attention")
                    and upsample_block.has_cross_attention
                ):
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        image_only_indicator=image_only_indicator,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        image_only_indicator=image_only_indicator,
                    )

            # 6. post-process
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
            sample = self.conv_out(sample)

            # 7. Reshape back to original shape
            # sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])
            # Rewrite for onediff SVD dynamic shape
            sample = sample.unflatten(0, shape=(batch_size, -1))

            if not return_dict:
                return (sample,)

            return UNetSpatioTemporalConditionOutput(sample=sample)
