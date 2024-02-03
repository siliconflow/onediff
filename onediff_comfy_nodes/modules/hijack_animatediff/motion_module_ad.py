# ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/animatediff/motion_module_ad.py
from einops import repeat
from ._config import animatediff_of, animatediff_pt
import oneflow as torch
from onediff.infer_compiler.transform import register

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
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch, channel, height, width = hidden_states.shape
        residual = hidden_states
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

        # hidden_states = super().forward(
        #     hidden_states,
        #     encoder_hidden_states,
        #     value=None,
        #     mask=attention_mask,
        #     scale_mask=scale_mask,
        # )
        from .motion_utils import CrossAttentionMM_OF

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
register(torch2oflow_class_map={VersatileAttention_PT_CLS: VersatileAttention_OF})
