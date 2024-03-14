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
import oneflow as torch

from packaging import version
import importlib.metadata

diffusers_0210_v = version.parse("0.21.0")
diffusers_version = version.parse(importlib.metadata.version("diffusers"))

transformed_diffusers = transform_mgr.transform_package("diffusers")

if diffusers_version >= diffusers_0210_v:

    class CrossAttnUpBlock2D(transformed_diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D):
        def forward(
            self,
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            output_like: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            enter_block_number: Optional[int] = None,
        ):
            # print("enter_block_number:", enter_block_number)
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
                    } if transformed_diffusers.utils.is_torch_version(">=", "1.11.0") else {}
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
                        hidden_states, upsample_size, output_like=output_like, scale=lora_scale
                    )

            return hidden_states, prv_f


    class UpBlock2D(transformed_diffusers.models.unet_2d_blocks.UpBlock2D):
        def forward(
            self,
            hidden_states,
            res_hidden_states_tuple,
            temb=None,
            upsample_size=None,
            output_like: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            enter_block_number: Optional[int] = None,
        ):
            # print("enter_block_number:", enter_block_number)
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

                    if transformed_diffusers.utils.is_torch_version(">=", "1.11.0"):
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
                    hidden_states = upsampler(hidden_states, upsample_size, scale=scale, output_like=output_like,)

            return hidden_states, prv_f
else:

    class CrossAttnUpBlock2D(transformed_diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D):
        def forward(
            self,
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            output_like: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            enter_block_number: Optional[int] = None,
        ):
            # print("enter_block_number:", enter_block_number)
            prv_f = []

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
                    } if transformed_diffusers.utils.is_torch_version(">=", "1.11.0") else {}
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
                    hidden_states = resnet(hidden_states, temb)
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
                        hidden_states, upsample_size, output_like
                    )

            return hidden_states, prv_f


    class UpBlock2D(transformed_diffusers.models.unet_2d_blocks.UpBlock2D):
        def forward(
            self,
            hidden_states,
            res_hidden_states_tuple,
            temb=None,
            upsample_size=None,
            output_like=None,
            enter_block_number: Optional[int] = None,
        ):
            # print("enter_block_number:", enter_block_number)
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

                    if transformed_diffusers.utils.is_torch_version(">=", "1.11.0"):
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
                    hidden_states = resnet(hidden_states, temb)

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size, output_like)

            return hidden_states, prv_f
