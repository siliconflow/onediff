import torch
import torch.nn as nn
from typing import Union, Optional, Tuple
from diffusers.utils import BaseOutput, logging
from oneflow.nn.graph.proxy import ProxyModule

from .unet_spatio_temporal_condition import UNetSpatioTemporalConditionOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FastUNetSpatioTemporalConditionModel(nn.Module):
    def __init__(self, unet_module):
        super().__init__()
        self.unet_module = unet_module

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_dict: bool = True,
        cache_features: Optional[torch.Tensor] = None,
        cache_branch: Optional[int] = None,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
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

        t_emb = self.unet_module.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.unet_module.time_embedding(t_emb)

        time_embeds = self.unet_module.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.unet_module.add_embedding(time_embeds)
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
        sample = self.unet_module.conv_in(sample)

        image_only_indicator = torch.zeros(
            batch_size, num_frames, dtype=sample.dtype, device=sample.device
        )

        # Branch: 4 down_blocks, each with 3 skip connections. Here we ignore the first skip branch, whose computations only has up_blocks but without down_blocks.
        if cache_branch is not None:
            each_module_num = len(self.unet_module.down_blocks[0].resnets) + 1
            down_cache_block_idx = cache_branch // each_module_num
            down_cache_module_idx = cache_branch % each_module_num

            up_cache_block_idx = (
                len(self.unet_module.up_blocks) - 1 - down_cache_block_idx
            )
            up_cache_module_idx = 1 - down_cache_module_idx
            if down_cache_module_idx == each_module_num - 1:
                up_cache_block_idx -= 1
                up_cache_module_idx = 2

        # 3. down
        down_block_res_samples = (sample,)
        for block_id, downsample_block in enumerate(self.unet_module.down_blocks):
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    exist_module_idx=down_cache_module_idx
                    if down_cache_block_idx == block_id
                    else None,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                    exist_module_idx=down_cache_module_idx
                    if down_cache_block_idx == block_id
                    else None,
                )

            down_block_res_samples += res_samples
            if down_cache_block_idx == block_id:
                break

        # 4. no mid
        sample = cache_features

        # 5. up
        for i, upsample_block in enumerate(self.unet_module.up_blocks):
            if i < up_cache_block_idx:
                continue

            if i == up_cache_block_idx:
                trunc_res_samples_len = (
                    len(upsample_block.resnets) - up_cache_module_idx
                )
            else:
                trunc_res_samples_len = len(upsample_block.resnets)

            res_samples = down_block_res_samples[-trunc_res_samples_len:]
            down_block_res_samples = down_block_res_samples[:-trunc_res_samples_len]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample, _ = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    enter_module_idx=up_cache_module_idx
                    if i == up_cache_block_idx
                    else None,
                )
            else:
                sample, _ = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                    enter_module_idx=up_cache_module_idx
                    if i == up_cache_block_idx
                    else None,
                )

        # 6. post-process
        sample = self.unet_module.conv_norm_out(sample)
        sample = self.unet_module.conv_act(sample)
        sample = self.unet_module.conv_out(sample)

        # 7. Reshape back to original shape
        if isinstance(self, ProxyModule):
            # Rewrite for onediff SVD dynamic shape
            sample = sample.unflatten(0, shape=(batch_size, -1))
        else:
            sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if not return_dict:
            return (sample, cache_features)

        return UNetSpatioTemporalConditionOutput(sample=sample)
