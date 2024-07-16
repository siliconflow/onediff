import torch
import torch as th
from modules import devices

cond_cast_unet = getattr(devices, "cond_cast_unet", lambda x: x)


# https://github.com/Mikubill/sd-webui-controlnet/blob/8bbbd0e55ef6e5d71b09c2de2727b36e7bc825b0/scripts/hook.py#L238
def aligned_adding(base, x, require_channel_alignment):
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

    return base + x.half()


class OnediffControlNetModel(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.time_embed = unet.time_embed
        self.input_blocks = unet.input_blocks
        self.label_emb = getattr(unet, "label_emb", None)
        self.middle_block = unet.middle_block
        self.output_blocks = unet.output_blocks
        self.out = unet.out
        self.model_channels = unet.model_channels
        # import ipdb; ipdb.set_trace()
        self.convert_to_fp16 = unet.convert_to_fp16.__get__(self)
        # print("something")

    @torch.autocast(device_type="cuda", enabled=False)
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

        # cast to half
        x = x.half()
        context = context.half()
        if y is not None:
            y = y.half()

        hs = []
        with th.no_grad():
            t_emb = cond_cast_unet(
                timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            )

            t_emb = t_emb.half()
            emb = self.time_embed(t_emb).half()

            if is_sdxl:
                assert y.shape[0] == x.shape[0]
                emb = emb + self.label_emb(y).half()

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
            h = th.cat(
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
            h = module(h, emb, context)

        # U-Net Output
        h = h.type(x.dtype)
        h = self.out(h)

        return h
