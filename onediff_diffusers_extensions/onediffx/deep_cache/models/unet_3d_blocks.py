import importlib.metadata
import types
from typing import Any, Dict, Optional, Tuple, Union

from packaging import version

diffusers_0260_v = version.parse("0.26.0")
diffusers_version = version.parse(importlib.metadata.version("diffusers"))

import diffusers
import torch
from diffusers.utils import is_torch_version


if diffusers_version >= diffusers_0260_v:
    from diffusers.models.unets import unet_3d_blocks as diffusers_unet_3d_blocks
else:
    from diffusers.models import unet_3d_blocks as diffusers_unet_3d_blocks


class DownBlockSpatioTemporal(diffusers_unet_3d_blocks.DownBlockSpatioTemporal):
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        exist_module_idx: Optional[int] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        # print("exist_module_idx:", exist_module_idx)
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
                        image_only_indicator,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        image_only_indicator,
                    )
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )

            output_states = output_states + (hidden_states,)

            if (
                exist_module_idx is not None
                and exist_module_idx == len(output_states) - 1
            ):
                return hidden_states, output_states

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnDownBlockSpatioTemporal(
    diffusers_unet_3d_blocks.CrossAttnDownBlockSpatioTemporal
):
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        exist_module_idx: Optional[int] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        # print("exist_module_idx:", exist_module_idx)
        output_states = ()

        blocks = list(zip(self.resnets, self.attentions))
        for resnet, attn in blocks:
            if self.training and self.gradient_checkpointing:  # TODO

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **ckpt_kwargs,
                )

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]

            output_states = output_states + (hidden_states,)
            if (
                exist_module_idx is not None
                and exist_module_idx == len(output_states) - 1
            ):
                return hidden_states, output_states

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class UpBlockSpatioTemporal(diffusers_unet_3d_blocks.UpBlockSpatioTemporal):
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        enter_module_idx: Optional[int] = None,
    ) -> torch.FloatTensor:
        # print("enter_module_idx:", enter_module_idx)
        prv_f = []
        for idx, resnet in enumerate(self.resnets):
            if enter_module_idx is not None and idx < enter_module_idx:
                continue

            prv_f.append(hidden_states)
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
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
                        image_only_indicator,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        image_only_indicator,
                    )
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states, prv_f


class CrossAttnUpBlockSpatioTemporal(
    diffusers_unet_3d_blocks.CrossAttnUpBlockSpatioTemporal
):
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        enter_module_idx: Optional[int] = None,
    ) -> torch.FloatTensor:
        # print("enter_module_idx:", enter_module_idx)
        prv_f = []
        for idx, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            if enter_module_idx is not None and idx < enter_module_idx:
                continue

            prv_f.append(hidden_states)
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:  # TODO

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **ckpt_kwargs,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states, prv_f


update_cls = {
    "DownBlockSpatioTemporal": DownBlockSpatioTemporal,
    "CrossAttnDownBlockSpatioTemporal": CrossAttnDownBlockSpatioTemporal,
    "UpBlockSpatioTemporal": UpBlockSpatioTemporal,
    "CrossAttnUpBlockSpatioTemporal": CrossAttnUpBlockSpatioTemporal,
}

if diffusers_version >= diffusers_0260_v:
    src_get_down_block = diffusers.models.unets.unet_3d_blocks.get_down_block
    src_get_up_block = diffusers.models.unets.unet_3d_blocks.get_up_block
else:
    src_get_down_block = diffusers.models.unet_3d_blocks.get_down_block
    src_get_up_block = diffusers.models.unet_3d_blocks.get_up_block

down_globals = {k: v for k, v in src_get_down_block.__globals__.items()}
down_globals.update(update_cls)
get_down_block = types.FunctionType(
    src_get_down_block.__code__, down_globals, argdefs=src_get_down_block.__defaults__
)


up_globals = {k: v for k, v in src_get_up_block.__globals__.items()}
up_globals.update(update_cls)
get_up_block = types.FunctionType(
    src_get_up_block.__code__, up_globals, argdefs=src_get_up_block.__defaults__
)
