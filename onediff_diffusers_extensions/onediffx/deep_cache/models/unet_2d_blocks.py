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
from typing import Any, Dict, Optional, Tuple

import types

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import diffusers
from diffusers.utils import is_torch_version, logging
from diffusers.models.activations import get_activation

if diffusers.__version__ >= "0.22.0":
    from diffusers.models.normalization import AdaGroupNorm
else:
    from diffusers.models.attention import AdaGroupNorm
from diffusers.models.attention_processor import (
    Attention,
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
)
from diffusers.models.dual_transformer_2d import DualTransformer2DModel
from diffusers.models.resnet import (
    Downsample2D,
    FirDownsample2D,
    FirUpsample2D,
    KDownsample2D,
    KUpsample2D,
    ResnetBlock2D,
    Upsample2D,
)
from diffusers.models.transformer_2d import Transformer2DModel

if diffusers.__version__ >= "0.26.0":
    from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D
else:
    from diffusers.models.unet_2d_blocks import DownEncoderBlock2D

from diffusers.models.unet_2d_blocks import (
    ResnetDownsampleBlock2D,
    AttnDownBlock2D,
    SimpleCrossAttnDownBlock2D,
    SkipDownBlock2D,
    AttnSkipDownBlock2D,
    AttnDownEncoderBlock2D,
    KDownBlock2D,
    KAttentionBlock,
    KCrossAttnDownBlock2D,
    ResnetUpsampleBlock2D,
    SimpleCrossAttnUpBlock2D,
    AttnUpBlock2D,
    SkipUpBlock2D,
    AttnSkipUpBlock2D,
    UpDecoderBlock2D,
    AttnUpDecoderBlock2D,
    KUpBlock2D,
    KCrossAttnUpBlock2D,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CrossAttnDownBlock2D(diffusers.models.unet_2d_blocks.CrossAttnDownBlock2D):
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        exist_block_number=None,
        additional_residuals=None,
    ):
        output_states = ()

        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )

        blocks = list(zip(self.resnets, self.attentions))

        for i, (resnet, attn) in enumerate(blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {
                    "use_reentrant": False
                } if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb, **ckpt_kwargs,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

            # apply additional residuals to the output of the last pair of resnet and attention blocks
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

            output_states = output_states + (hidden_states,)
            if (
                exist_block_number is not None
                and len(output_states) == exist_block_number + 1
            ):
                return hidden_states, output_states

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale=lora_scale)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class DownBlock2D(diffusers.models.unet_2d_blocks.DownBlock2D):
    def forward(
        self, hidden_states, temb=None, scale: float = 1.0, exist_block_number=None,
    ):
        output_states = ()

        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                hidden_states = resnet(hidden_states, temb, scale=scale)

            output_states = output_states + (hidden_states,)
            if (
                exist_block_number is not None
                and len(output_states) == exist_block_number + 1
            ):
                return hidden_states, output_states

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale=scale)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlock2D(diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D):
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        enter_block_number: Optional[int] = None,
    ):
        prv_f = []
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )

        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            # pop res hidden states

            if (
                enter_block_number is not None
                and i < len(self.resnets) - enter_block_number - 1
            ):
                continue

            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            prv_f.append(hidden_states)
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {
                    "use_reentrant": False
                } if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb, **ckpt_kwargs,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(
                    hidden_states, upsample_size, scale=lora_scale
                )

        return hidden_states, prv_f


class UpBlock2D(diffusers.models.unet_2d_blocks.UpBlock2D):
    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        upsample_size=None,
        scale: float = 1.0,
        enter_block_number: Optional[int] = None,
    ):
        prv_f = []

        for idx, resnet in enumerate(self.resnets):

            if (
                enter_block_number is not None
                and idx < len(self.resnets) - enter_block_number - 1
            ):
                continue

            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            prv_f.append(hidden_states)
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                hidden_states = resnet(hidden_states, temb, scale=scale)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size, scale=scale)

        return hidden_states, prv_f


update_cls = {
  "CrossAttnDownBlock2D": CrossAttnDownBlock2D,
  "DownBlock2D": DownBlock2D,
  "CrossAttnUpBlock2D": CrossAttnUpBlock2D,
  "UpBlock2D": UpBlock2D,
}

src_get_down_block = diffusers.models.unet_2d_blocks.get_down_block
down_globals = {k : v for k, v in src_get_down_block.__globals__.items()}
down_globals.update(update_cls)

get_down_block = types.FunctionType(src_get_down_block.__code__, down_globals, argdefs=src_get_down_block.__defaults__)

src_get_up_block = diffusers.models.unet_2d_blocks.get_up_block
up_globals = {k : v for k, v in src_get_up_block.__globals__.items()}
up_globals.update(update_cls)
get_up_block = types.FunctionType(src_get_up_block.__code__, up_globals, argdefs=src_get_up_block.__defaults__)
