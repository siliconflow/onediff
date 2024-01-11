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

from typing import Optional, Tuple, Union


class TemporalDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1
        )
        self.mid_block = MidBlockTemporalDecoder(
            num_layers=self.layers_per_block,
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            attention_head_dim=block_out_channels[-1],
        )

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            up_block = UpBlockTemporalDecoder(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=32, eps=1e-6
        )

        self.conv_act = nn.SiLU()
        self.conv_out = torch.nn.Conv2d(
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        conv_out_kernel_size = (3, 1, 1)
        padding = [int(k // 2) for k in conv_out_kernel_size]
        self.time_conv_out = torch.nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=conv_out_kernel_size,
            padding=padding,
        )

        self.gradient_checkpointing = False

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
                    create_custom_forward(self.mid_block), sample, image_only_indicator,
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block), sample, image_only_indicator,
                    )
        else:
            # middle
            sample = self.mid_block(sample, image_only_indicator=image_only_indicator)
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
        sample = (
            sample[None, :]
            .reshape(batch_size, -1, channels, height, width)
            .permute(0, 2, 1, 3, 4)
        )
        sample = self.time_conv_out(sample)

        # sample = sample.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
        # Dynamic shape for VAE divide chunks
        sample = sample.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)

        return sample


# VideoResBlock
class SpatioTemporalResBlock(nn.Module):
    r"""
    A SpatioTemporal Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the spatial resenet.
        temporal_eps (`float`, *optional*, defaults to `eps`): The epsilon to use for the temporal resnet.
        merge_factor (`float`, *optional*, defaults to `0.5`): The merge factor to use for the temporal mixing.
        merge_strategy (`str`, *optional*, defaults to `learned_with_images`):
            The merge strategy to use for the temporal mixing.
        switch_spatial_to_temporal_mix (`bool`, *optional*, defaults to `False`):
            If `True`, switch the spatial and temporal mixing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
        temporal_eps: Optional[float] = None,
        merge_factor: float = 0.5,
        merge_strategy="learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
    ):
        super().__init__()

        self.spatial_res_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=eps,
        )

        self.temporal_res_block = TemporalResnetBlock(
            in_channels=out_channels if out_channels is not None else in_channels,
            out_channels=out_channels if out_channels is not None else in_channels,
            temb_channels=temb_channels,
            eps=temporal_eps if temporal_eps is not None else eps,
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor,
            merge_strategy=merge_strategy,
            switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,
        )

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
        hidden_states_mix = (
            hidden_states[None, :]
            .reshape(batch_size, -1, channels, height, width)
            .permute(0, 2, 1, 3, 4)
        )
        hidden_states = (
            hidden_states[None, :]
            .reshape(batch_size, -1, channels, height, width)
            .permute(0, 2, 1, 3, 4)
        )

        if temb is not None:
            temb = temb.reshape(batch_size, num_frames, -1)

        hidden_states = self.temporal_res_block(hidden_states, temb)
        hidden_states = self.time_mixer(
            x_spatial=hidden_states_mix,
            x_temporal=hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
        # Dynamic shape for VAE divide chunks
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            -1, channels, height, width
        )
        return hidden_states
