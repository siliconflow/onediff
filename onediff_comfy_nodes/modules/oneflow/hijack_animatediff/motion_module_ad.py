# ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/animatediff/motion_module_ad.py
import oneflow as torch
from einops import repeat
from onediff.infer_compiler.backends.oneflow.transform import register

from ._config import animatediff_of, animatediff_pt

TemporalTransformer3DModel_OF_CLS = (
    animatediff_of.animatediff.motion_module_ad.TemporalTransformer3DModel
)
TemporalTransformer3DModel_PT_CLS = (
    animatediff_pt.animatediff.motion_module_ad.TemporalTransformer3DModel
)
VersatileAttention_OF_CLS = (
    animatediff_of.animatediff.motion_module_ad.VersatileAttention
)
VersatileAttention_PT_CLS = (
    animatediff_pt.animatediff.motion_module_ad.VersatileAttention
)


class TemporalTransformer3DModel_OF(TemporalTransformer3DModel_OF_CLS):
    def get_cameractrl_effect(self, hidden_states: torch.Tensor):
        # if no raw camera_Ctrl, return None
        if self.raw_cameractrl_effect is None:
            return 1.0
        # if raw_cameractrl is not a Tensor, return it (should be a float)
        if type(self.raw_cameractrl_effect) != torch.Tensor:
            return self.raw_cameractrl_effect
        shape = hidden_states.shape
        batch, channel, height, width = shape
        # if temp_cameractrl already calculated, return it
        if self.temp_cameractrl_effect != None:
            # check if hidden_states batch matches
            if batch == self.prev_cameractrl_hidden_states_batch:
                if self.sub_idxs is not None:
                    return self.temp_cameractrl_effect[:, self.sub_idxs, :]
                return self.temp_cameractrl_effect
            # if does not match, reset cached temp_cameractrl and recalculate it
            del self.temp_cameractrl_effect
            self.temp_cameractrl_effect = None
        # otherwise, calculate temp_cameractrl
        self.prev_cameractrl_hidden_states_batch = batch
        mask = prepare_mask_batch(
            self.raw_scale_mask, shape=(self.full_length, 1, height, width)
        )
        mask = repeat_to_batch_size(mask, self.full_length)
        # if mask not the same amount length as full length, make it match
        if self.full_length != mask.shape[0]:
            mask = broadcast_image_to(mask, self.full_length, 1)
        # reshape mask to attention K shape (h*w, latent_count, 1)
        batch, channel, height, width = mask.shape
        # first, perform same operations as on hidden_states,
        # turning (b, c, h, w) -> (b, h*w, c)
        mask = mask.permute(0, 2, 3, 1).reshape(batch, height * width, channel)
        # then, make it the same shape as attention's k, (h*w, b, c)
        mask = mask.permute(1, 0, 2)
        # make masks match the expected length of h*w
        batched_number = shape[0] // self.video_length
        if batched_number > 1:
            mask = torch.cat([mask] * batched_number, dim=0)
        # cache mask and set to proper device
        self.temp_cameractrl_effect = mask
        # move temp_cameractrl to proper dtype + device
        self.temp_cameractrl_effect = self.temp_cameractrl_effect.to(
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        # return subset of masks, if needed
        if self.sub_idxs is not None:
            return self.temp_cameractrl_effect[:, self.sub_idxs, :]
        return self.temp_cameractrl_effect

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        view_options=None,
        mm_kwargs: dict[str] = None,
    ):
        batch, channel, height, width = hidden_states.shape
        residual = hidden_states
        cameractrl_effect = self.get_cameractrl_effect(hidden_states)

        scale_mask = self.get_scale_mask(hidden_states)
        # add some casts for fp8 purposes - does not affect speed otherwise
        hidden_states = self.norm(hidden_states).to(hidden_states.dtype)
        inner_dim = hidden_states.shape[1]
        # hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
        #     batch, height * width, inner_dim
        # )
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, -1, inner_dim)
        hidden_states = self.proj_in(hidden_states).to(hidden_states.dtype)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                video_length=self.video_length,
                scale_mask=scale_mask,
                cameractrl_effect=cameractrl_effect,
                view_options=view_options,
                mm_kwargs=mm_kwargs,
            )

        # output
        hidden_states = self.proj_out(hidden_states)
        # hidden_states = (
        #     hidden_states.reshape(batch, height, width, inner_dim)
        #     .permute(0, 3, 1, 2)
        #     .contiguous()
        # )
        # b (h w) i -> b i h w
        hidden_states = hidden_states.permute(0, 2, 1).reshape_as(residual)

        output = hidden_states + residual

        return output


class VersatileAttention_OF(VersatileAttention_OF_CLS):
    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        scale_mask=None,
        cameractrl_effect=1.0,
        mm_kwargs: dict[str] = {},
    ):
        if self.attention_mode != "Temporal":
            raise NotImplementedError
        d = hidden_states.shape[1]
        # hidden_states = rearrange(
        #     hidden_states, "(b f) d c -> (b d) f c", f=video_length
        # )
        # (b f) d c -> b f d c -> b d f c -> (b d) f c
        # print(f'in forward: hidden_states.shape: {hidden_states.shape}')
        b = hidden_states.shape[0] // video_length
        hidden_states = (
            hidden_states.unflatten(0, (b, -1)).permute(0, 2, 1, 3).flatten(0, 1)
        )

        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states).to(hidden_states.dtype)

        encoder_hidden_states = (
            repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
            if encoder_hidden_states is not None
            else encoder_hidden_states
        )
        if (
            self.camera_feature_enabled
            and self.qkv_merge is not None
            and mm_kwargs is not None
            and "camera_feature" in mm_kwargs
        ):
            camera_feature: torch.Tensor = mm_kwargs["camera_feature"]
            hidden_states = (
                self.qkv_merge(hidden_states + camera_feature) + hidden_states
            ) * cameractrl_effect + hidden_states * (1.0 - cameractrl_effect)

        # hidden_states = super().forward(
        #     hidden_states,
        #     encoder_hidden_states,
        #     value=None,
        #     mask=attention_mask,
        #     scale_mask=scale_mask,
        # )
        from .utils_motion import CrossAttentionMM_OF

        hidden_states = CrossAttentionMM_OF.forward(
            self,
            hidden_states,
            encoder_hidden_states,
            value=None,
            mask=attention_mask,
            scale_mask=scale_mask,
        )

        # hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
        # (b d) f c -> b d f c -> b f d c -> (b f) d c
        hidden_states = (
            hidden_states.unflatten(0, (b, -1)).permute(0, 2, 1, 3).flatten(0, 1)
        )

        return hidden_states


register(
    torch2oflow_class_map={
        TemporalTransformer3DModel_PT_CLS: TemporalTransformer3DModel_OF
    }
)

# import torch as torch_pt
# from onediff.infer_compiler.backends.oneflow.transform import torch2oflow

# @torch2oflow.register(TemporalTransformer3DModel_PT_CLS)
# def _(mod, verbose=False):
#     of_mod = torch2oflow.dispatch(torch_pt.nn.Module)(mod, verbose)
#     of_mod.video_length = torch.tensor(mod.video_length)
#     return of_mod

register(torch2oflow_class_map={VersatileAttention_PT_CLS: VersatileAttention_OF})
