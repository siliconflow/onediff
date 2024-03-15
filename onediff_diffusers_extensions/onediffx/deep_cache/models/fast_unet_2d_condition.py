import torch
import torch.nn as nn
from typing import Union, Optional, Dict, Any, Tuple, List

from packaging import version
import importlib.metadata
from oneflow.nn.graph.proxy import ProxyModule

diffusers_0210_v = version.parse("0.21.0")
diffusers_0270_v = version.parse("0.27.0")
diffusers_version = version.parse(importlib.metadata.version("diffusers"))

from diffusers.utils import BaseOutput, logging

from .unet_2d_condition import UNet2DConditionModel
from .unet_2d_condition import UNet2DConditionOutput

try:
    USE_PEFT_BACKEND = diffusers.utils.USE_PEFT_BACKEND
    scale_lora_layers = diffusers.utils.scale_lora_layers
    unscale_lora_layers = diffusers.utils.unscale_lora_layers
except Exception as e:
    USE_PEFT_BACKEND = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FastUNet2DConditionModel(nn.Module):
    def __init__(self, unet_module):
        super().__init__()
        self.unet_module = unet_module

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        replicate_prv_feature: Optional[List[torch.Tensor]] = None,
        cache_layer_id: Optional[int] = None,
        cache_block_id: Optional[int] = None,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttnProcessor`].
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containin additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_2d_condition.UNet2DConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2 ** self.unet_module.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        # forward_upsample_size = False
        # interpolate through upsample_size
        forward_upsample_size = True
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(sample.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.unet_module.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        if diffusers_version < diffusers_0270_v:
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
            # modified to support dynamic shape for onediff
            if isinstance(self, ProxyModule):
                timesteps = torch._C.broadcast_dim_like(timesteps, sample, dim=0)
            else:
                timesteps = timesteps.expand(sample.shape[0])

            t_emb = self.unet_module.time_proj(timesteps)

            # `Timesteps` does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=sample.dtype)
        else:
            t_emb = self.unet_module.get_time_embed(sample=sample, timestep=timestep)

        emb = self.unet_module.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if diffusers_version < diffusers_0270_v:
            if self.unet_module.class_embedding is not None:
                if class_labels is None:
                    raise ValueError(
                        "class_labels should be provided when num_class_embeds > 0"
                    )

                if self.unet_module.config.class_embed_type == "timestep":
                    class_labels = self.unet_module.time_proj(class_labels)

                    # `Timesteps` does not contain any weights and will always return f32 tensors
                    # there might be better ways to encapsulate this.
                    class_labels = class_labels.to(dtype=sample.dtype)

                class_emb = self.unet_module.class_embedding(class_labels).to(
                    dtype=sample.dtype
                )

                if self.unet_module.config.class_embeddings_concat:
                    emb = torch.cat([emb, class_emb], dim=-1)
                else:
                    emb = emb + class_emb

            if self.unet_module.config.addition_embed_type == "text":
                aug_emb = self.unet_module.add_embedding(encoder_hidden_states)
            elif self.unet_module.config.addition_embed_type == "text_image":
                # Kandinsky 2.1 - style
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.unet_module.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                    )

                image_embs = added_cond_kwargs.get("image_embeds")
                text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
                aug_emb = self.unet_module.add_embedding(text_embs, image_embs)
            elif self.unet_module.config.addition_embed_type == "text_time":
                # SDXL - style
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.unet_module.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                    )
                text_embeds = added_cond_kwargs.get("text_embeds")
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.unet_module.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                    )
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.unet_module.add_time_proj(time_ids.flatten())
                # modified to support dynamic shape for onediff
                if isinstance(self, ProxyModule):
                    time_embeds = time_embeds.unflatten(
                        dim=0, shape=(-1, time_ids.shape[1])
                    ).flatten(1, 2)
                else:
                    time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.unet_module.add_embedding(add_embeds)
            elif self.unet_module.config.addition_embed_type == "image":
                # Kandinsky 2.2 - style
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.unet_module.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                    )
                image_embs = added_cond_kwargs.get("image_embeds")
                aug_emb = self.unet_module.add_embedding(image_embs)
            elif self.unet_module.config.addition_embed_type == "image_hint":
                # Kandinsky 2.2 - style
                if (
                    "image_embeds" not in added_cond_kwargs
                    or "hint" not in added_cond_kwargs
                ):
                    raise ValueError(
                        f"{self.unet_module.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                    )
                image_embs = added_cond_kwargs.get("image_embeds")
                hint = added_cond_kwargs.get("hint")
                aug_emb, hint = self.unet_module.add_embedding(image_embs, hint)
                sample = torch.cat([sample, hint], dim=1)
        else:
            aug_emb = self.unet_module.get_aug_embed(
                emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
            )
            if self.unet_module.config.addition_embed_type == "image_hint":
                aug_emb, hint = aug_emb
                sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.unet_module.time_embed_act is not None:
            emb = self.unet_module.time_embed_act(emb)

        if diffusers_version < diffusers_0270_v:
            if (
                self.unet_module.encoder_hid_proj is not None
                and self.unet_module.config.encoder_hid_dim_type == "text_proj"
            ):
                encoder_hidden_states = self.unet_module.encoder_hid_proj(
                    encoder_hidden_states
                )
            elif (
                self.unet_module.encoder_hid_proj is not None
                and self.unet_module.config.encoder_hid_dim_type == "text_image_proj"
            ):
                # Kadinsky 2.1 - style
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.unet_module.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                    )

                image_embeds = added_cond_kwargs.get("image_embeds")
                encoder_hidden_states = self.unet_module.encoder_hid_proj(
                    encoder_hidden_states, image_embeds
                )
            elif (
                self.unet_module.encoder_hid_proj is not None
                and self.unet_module.config.encoder_hid_dim_type == "image_proj"
            ):
                # Kandinsky 2.2 - style
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.unet_module.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                    )
                image_embeds = added_cond_kwargs.get("image_embeds")
                encoder_hidden_states = self.unet_module.encoder_hid_proj(image_embeds)
            elif (
                self.unet_module.encoder_hid_proj is not None
                and self.unet_module.config.encoder_hid_dim_type == "ip_image_proj"
            ):
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.unet_module.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                    )
                image_embeds = added_cond_kwargs.get("image_embeds")
                image_embeds = self.unet_module.encoder_hid_proj(image_embeds).to(
                    encoder_hidden_states.dtype
                )
                encoder_hidden_states = torch.cat(
                    [encoder_hidden_states, image_embeds], dim=1
                )
        else:
            encoder_hidden_states = self.unet_module.process_encoder_hidden_states(
                encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
            )
        # 2. pre-process
        sample = self.unet_module.conv_in(sample)

        # 2.5 GLIGEN position net
        if (
            cross_attention_kwargs is not None
            and cross_attention_kwargs.get("gligen", None) is not None
        ):
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {
                "objs": self.unet_module.position_net(**gligen_args)
            }

        # 3. down
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )
        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        is_controlnet = (
            mid_block_additional_residual is not None
            and down_block_additional_residuals is not None
        )
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None
        # maintain backward compatibility for legacy usage, where
        #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
        #       but can only use one or the other
        if (
            not is_adapter
            and mid_block_additional_residual is None
            and down_block_additional_residuals is not None
        ):
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                       and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                       for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)

        for i, downsample_block in enumerate(self.unet_module.down_blocks):
            if i > cache_layer_id:
                break

            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals[
                        "additional_residuals"
                    ] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    exist_block_number=cache_block_id if i == cache_layer_id else None,
                    **additional_residuals,
                )
            else:
                if diffusers_version < diffusers_0210_v or diffusers_version >= diffusers_0270_v:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        exist_block_number=cache_block_id if i == cache_layer_id else None,
                    )
                else:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        scale=lora_scale,
                        exist_block_number=cache_block_id if i == cache_layer_id else None,
                    )
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = (
                    down_block_res_sample + down_block_additional_residual
                )
                new_down_block_res_samples = new_down_block_res_samples + (
                    down_block_res_sample,
                )

            down_block_res_samples = new_down_block_res_samples
        
        # No Middle
        # Up
        # print("down_block_res_samples:", [res_sample.shape for res_sample in down_block_res_samples])
        sample = replicate_prv_feature
        max_block_depth = (
            len(self.unet_module.down_blocks[cache_layer_id].attentions)
            if hasattr(self.unet_module.down_blocks[cache_layer_id], "attentions")
            else len(self.unet_module.down_blocks[cache_layer_id].resnets)
        )
        if cache_block_id == max_block_depth:
            cache_block_id = 0
            cache_layer_id += 1
        else:
            cache_block_id += 1

        for i, upsample_block in enumerate(self.unet_module.up_blocks):
            if i < len(self.unet_module.up_blocks) - 1 - cache_layer_id:
                continue

            if i == len(self.unet_module.up_blocks) - 1 - cache_layer_id:
                trunc_upsample_block = cache_block_id + 1
            else:
                trunc_upsample_block = len(upsample_block.resnets)

            is_final_block = i == len(self.unet_module.up_blocks) - 1

            res_samples = down_block_res_samples[-trunc_upsample_block:]
            down_block_res_samples = down_block_res_samples[:-trunc_upsample_block]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            # To support dynamic switching of special resolutions, pass a like tensor.
            output_like = None
            if not is_final_block:
                output_like = down_block_res_samples[-1]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                # print(sample.shape, [res_sample.shape for res_sample in res_samples])
                sample, _ = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    output_like=output_like,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    enter_block_number=cache_block_id
                    if i == len(self.unet_module.up_blocks) - 1 - cache_layer_id
                    else None,
                )
            else:
                if diffusers_version < diffusers_0210_v or diffusers_version >= diffusers_0270_v:
                    sample, _ = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                        output_like=output_like,
                        enter_block_number=cache_block_id
                        if i == len(self.unet_module.up_blocks) - 1 - cache_layer_id
                        else None,
                    )
                else:
                    sample, _ = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                        output_like=output_like,
                        scale=lora_scale,
                        enter_block_number=cache_block_id
                        if i == len(self.unet_module.up_blocks) - 1 - cache_layer_id
                        else None,
                    )

        prv_f = replicate_prv_feature

        # 6. post-process
        if self.unet_module.conv_norm_out:
            sample = self.unet_module.conv_norm_out(sample)
            sample = self.unet_module.conv_act(sample)
        sample = self.unet_module.conv_out(sample)
        if not return_dict:
            return (
                sample,
                prv_f,
            )
        return UNet2DConditionOutput(sample=sample)
