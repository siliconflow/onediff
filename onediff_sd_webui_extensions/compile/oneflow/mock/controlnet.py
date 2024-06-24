import oneflow as flow
import torch
import torch as th
from compile.oneflow.mock.common import (
    CrossAttentionOflow,
    GroupNorm32Oflow,
    timestep_embedding,
)
from ldm.modules.attention import CrossAttention
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.diffusionmodules.util import GroupNorm32
from modules import devices

from onediff.infer_compiler.backends.oneflow.transform import proxy_class

cond_cast_unet = getattr(devices, "cond_cast_unet", lambda x: x)


# https://github.com/Mikubill/sd-webui-controlnet/blob/8bbbd0e55ef6e5d71b09c2de2727b36e7bc825b0/scripts/hook.py#L238
def torch_aligned_adding(base, x, require_channel_alignment):
    if isinstance(x, float):
        if x == 0.0:
            return base
        return base + x

    if require_channel_alignment:
        zeros = torch.zeros_like(base)
        zeros[:, : x.shape[1], ...] = x
        x = zeros

    # resize to sample resolution
    base_h, base_w = base.shape[-2:]
    xh, xw = x.shape[-2:]

    if xh > 1 or xw > 1:
        if base_h != xh or base_w != xw:
            # logger.info('[Warning] ControlNet finds unexpected mis-alignment in tensor shape.')
            x = th.nn.functional.interpolate(x, size=(base_h, base_w), mode="nearest")

    return base + x


# Due to the tracing mechanism in OneFlow, it's crucial to ensure that
# the same conditional branches are taken during the first run as in subsequent runs.
# Therefore, certain "optimizations" have been modified.
def oneflow_aligned_adding(base, x, require_channel_alignment):
    if isinstance(x, float):
        # remove `if x == 0.0: return base` here
        return base + x

    if require_channel_alignment:
        zeros = flow.zeros_like(base)
        zeros[:, : x.shape[1], ...] = x
        x = zeros

    # resize to sample resolution
    base_h, base_w = base.shape[-2:]
    xh, xw = x.shape[-2:]

    if xh > 1 or xw > 1 and (base_h != xh or base_w != xw):
        # logger.info('[Warning] ControlNet finds unexpected mis-alignment in tensor shape.')
        x = flow.nn.functional.interpolate(x, size=(base_h, base_w), mode="nearest")
    return base + x


class TorchOnediffControlNetModel(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.time_embed = unet.time_embed
        self.input_blocks = unet.input_blocks
        self.label_emb = getattr(unet, "label_emb", None)
        self.middle_block = unet.middle_block
        self.output_blocks = unet.output_blocks
        self.out = unet.out
        self.model_channels = unet.model_channels

    def forward(
        self,
        x,
        timesteps,
        context,
        y,
        total_t2i_adapter_embedding,
        total_controlnet_embedding,
        is_sdxl,
        require_inpaint_hijack,
    ):
        from ldm.modules.diffusionmodules.util import timestep_embedding

        hs = []
        with th.no_grad():
            t_emb = cond_cast_unet(
                timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            )
            emb = self.time_embed(t_emb)

            if is_sdxl:
                assert y.shape[0] == x.shape[0]
                emb = emb + self.label_emb(y)

            h = x
            for i, module in enumerate(self.input_blocks):
                self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
                h = module(h, emb, context)

                t2i_injection = [3, 5, 8] if is_sdxl else [2, 5, 8, 11]

                if i in t2i_injection:
                    h = torch_aligned_adding(
                        h, total_t2i_adapter_embedding.pop(0), require_inpaint_hijack
                    )

                hs.append(h)

            self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
            h = self.middle_block(h, emb, context)

        # U-Net Middle Block
        h = torch_aligned_adding(
            h, total_controlnet_embedding.pop(), require_inpaint_hijack
        )

        if len(total_t2i_adapter_embedding) > 0 and is_sdxl:
            h = torch_aligned_adding(
                h, total_t2i_adapter_embedding.pop(0), require_inpaint_hijack
            )

        # U-Net Decoder
        for i, module in enumerate(self.output_blocks):
            self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
            h = th.cat(
                [
                    h,
                    torch_aligned_adding(
                        hs.pop(),
                        total_controlnet_embedding.pop(),
                        require_inpaint_hijack,
                    ),
                ],
                dim=1,
            )
            h = module(h, emb, context)

        # U-Net Output
        h = h.type(x.dtype)
        h = self.out(h)

        return h


class OneFlowOnediffControlNetModel(proxy_class(UNetModel)):
    def forward(
        self,
        x,
        timesteps,
        context,
        y,
        total_t2i_adapter_embedding,
        total_controlnet_embedding,
        is_sdxl,
        require_inpaint_hijack,
    ):
        x = x.half()
        if y is not None:
            y = y.half()
        context = context.half()
        hs = []
        with flow.no_grad():
            t_emb = cond_cast_unet(
                timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            )
            emb = self.time_embed(t_emb.half())

            if is_sdxl:
                assert y.shape[0] == x.shape[0]
                emb = emb + self.label_emb(y)

            h = x
            for i, module in enumerate(self.input_blocks):
                self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
                h = module(h, emb, context)

                t2i_injection = [3, 5, 8] if is_sdxl else [2, 5, 8, 11]

                if i in t2i_injection:
                    h = oneflow_aligned_adding(
                        h, total_t2i_adapter_embedding.pop(0), require_inpaint_hijack
                    )

                hs.append(h)

            self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
            h = self.middle_block(h, emb, context)

        # U-Net Middle Block
        h = oneflow_aligned_adding(
            h, total_controlnet_embedding.pop(), require_inpaint_hijack
        )

        if len(total_t2i_adapter_embedding) > 0 and is_sdxl:
            h = oneflow_aligned_adding(
                h, total_t2i_adapter_embedding.pop(0), require_inpaint_hijack
            )

        # U-Net Decoder
        for i, module in enumerate(self.output_blocks):
            self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
            h = flow.cat(
                [
                    h,
                    oneflow_aligned_adding(
                        hs.pop(),
                        total_controlnet_embedding.pop(),
                        require_inpaint_hijack,
                    ),
                ],
                dim=1,
            )
            h = h.half()
            h = module(h, emb, context)

        # U-Net Output
        h = h.type(x.dtype)
        h = self.out(h)

        return h


torch2oflow_class_map = {
    CrossAttention: CrossAttentionOflow,
    GroupNorm32: GroupNorm32Oflow,
    TorchOnediffControlNetModel: OneFlowOnediffControlNetModel,
}
# register(package_names=["scripts.hook"], torch2oflow_class_map=torch2oflow_class_map)
