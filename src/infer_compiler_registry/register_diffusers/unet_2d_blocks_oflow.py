import importlib.metadata
from typing import Any, Dict, List, Optional, Tuple, Union

import oneflow as torch
from onediff.infer_compiler.backends.oneflow.transform import transform_mgr
from packaging import version

diffusers_0210_v = version.parse("0.21.0")
diffusers_0260_v = version.parse("0.26.0")
diffusers_version = version.parse(importlib.metadata.version("diffusers"))

transformed_diffusers = transform_mgr.transform_package("diffusers")

if diffusers_version >= version.parse("0.26.0"):
    diffusers_unet_2d_blocks = transformed_diffusers.models.unets.unet_2d_blocks
else:
    diffusers_unet_2d_blocks = transformed_diffusers.models.unet_2d_blocks

if diffusers_version < diffusers_0210_v:

    class AttnUpBlock2D(diffusers_unet_2d_blocks.AttnUpBlock2D):
        def forward(
            self,
            hidden_states,
            res_hidden_states_tuple,
            temb=None,
            upsample_size=None,
            output_like=None,
        ):
            for resnet, attn in zip(self.resnets, self.attentions):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states)

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    if self.upsample_type == "resnet":
                        hidden_states = upsampler(hidden_states, temb=temb)
                    else:
                        hidden_states = upsampler(hidden_states)

    class CrossAttnUpBlock2D(diffusers_unet_2d_blocks.CrossAttnUpBlock2D):
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
        ):
            for resnet, attn in zip(self.resnets, self.attentions):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = (
                        {"use_reentrant": False}
                        if transformed_diffusers.utils.is_torch_version(">=", "1.11.0")
                        else {}
                    )
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
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
                    hidden_states = upsampler(hidden_states, upsample_size, output_like)

            return hidden_states

    class UpBlock2D(diffusers_unet_2d_blocks.UpBlock2D):
        def forward(
            self,
            hidden_states,
            res_hidden_states_tuple,
            temb=None,
            upsample_size=None,
            output_like=None,
        ):
            for resnet in self.resnets:
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
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

            return hidden_states

else:

    class AttnUpBlock2D(diffusers_unet_2d_blocks.AttnUpBlock2D):
        def forward(
            self,
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            upsample_size: Optional[int] = None,
            output_like: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
        ) -> torch.FloatTensor:
            for resnet, attn in zip(self.resnets, self.attentions):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                hidden_states = resnet(hidden_states, temb, scale=scale)
                cross_attention_kwargs = {"scale": scale}
                hidden_states = attn(hidden_states, **cross_attention_kwargs)

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    if self.upsample_type == "resnet":
                        hidden_states = upsampler(hidden_states, temb=temb, scale=scale)
                    else:
                        hidden_states = upsampler(hidden_states, scale=scale)

            return hidden_states

    class CrossAttnUpBlock2D(diffusers_unet_2d_blocks.CrossAttnUpBlock2D):
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
        ) -> torch.FloatTensor:
            lora_scale = (
                cross_attention_kwargs.get("scale", 1.0)
                if cross_attention_kwargs is not None
                else 1.0
            )
            is_freeu_enabled = (
                getattr(self, "s1", None)
                and getattr(self, "s2", None)
                and getattr(self, "b1", None)
                and getattr(self, "b2", None)
            )

            for resnet, attn in zip(self.resnets, self.attentions):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                # FreeU: Only operate on the first two stages
                if is_freeu_enabled:
                    hidden_states, res_hidden_states = apply_freeu(
                        self.resolution_idx,
                        hidden_states,
                        res_hidden_states,
                        s1=self.s1,
                        s2=self.s2,
                        b1=self.b1,
                        b2=self.b2,
                    )

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = (
                        {"use_reentrant": False}
                        if transformed_diffusers.utils.is_torch_version(">=", "1.11.0")
                        else {}
                    )
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
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
                        hidden_states,
                        upsample_size,
                        output_like=output_like,
                        scale=lora_scale,
                    )

            return hidden_states

    class UpBlock2D(diffusers_unet_2d_blocks.UpBlock2D):
        def forward(
            self,
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            upsample_size: Optional[int] = None,
            output_like: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
        ) -> torch.FloatTensor:
            is_freeu_enabled = (
                getattr(self, "s1", None)
                and getattr(self, "s2", None)
                and getattr(self, "b1", None)
                and getattr(self, "b2", None)
            )

            for resnet in self.resnets:
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                # FreeU: Only operate on the first two stages
                if is_freeu_enabled:
                    hidden_states, res_hidden_states = apply_freeu(
                        self.resolution_idx,
                        hidden_states,
                        res_hidden_states,
                        s1=self.s1,
                        s2=self.s2,
                        b1=self.b1,
                        b2=self.b2,
                    )

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
                    hidden_states = upsampler(
                        hidden_states,
                        upsample_size,
                        output_like=output_like,
                        scale=scale,
                    )

            return hidden_states
