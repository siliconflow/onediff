import oneflow as flow  # usort: skip
from compile.oneflow.mock.common import timestep_embedding
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from modules import devices

from onediff.infer_compiler.backends.oneflow.transform import proxy_class

cond_cast_unet = getattr(devices, "cond_cast_unet", lambda x: x)


# Due to the tracing mechanism in OneFlow, it's crucial to ensure that
# the same conditional branches are taken during the first run as in subsequent runs.
# Therefore, certain "optimizations" have been modified.
def aligned_adding(base, x, require_channel_alignment):
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


def cat(tensors, *args, **kwargs):
    if len(tensors) == 2:
        a, b = tensors
        a = flow.nn.functional.interpolate_like(a, like=b, mode="nearest")
        tensors = (a, b)
    return flow.cat(tensors, *args, **kwargs)


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
                    h = aligned_adding(
                        h, total_t2i_adapter_embedding.pop(0), require_inpaint_hijack
                    )

                hs.append(h)

            self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
            h = self.middle_block(h, emb, context)

        # U-Net Middle Block
        h = aligned_adding(h, total_controlnet_embedding.pop(), require_inpaint_hijack)

        if len(total_t2i_adapter_embedding) > 0 and is_sdxl:
            h = aligned_adding(
                h, total_t2i_adapter_embedding.pop(0), require_inpaint_hijack
            )

        # U-Net Decoder
        for i, module in enumerate(self.output_blocks):
            self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
            h = cat(
                [
                    h,
                    aligned_adding(
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
