import onediff_shared
import torch

from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.openaimodel import ResBlock, UNetModel
from ldm.modules.diffusionmodules.util import timestep_embedding
from modules.sd_hijack_utils import CondFunc
from modules.sd_hijack import apply_optimizations
from onediff_utils import singleton_decorator

from onediff.infer_compiler import compile


def nexfort_compile_ldm_unet(unet_model, *, options=None):
    create_cond_func()
    apply_optimizations("sdp-no-mem - scaled dot product without memory efficient attention")
    if not isinstance(unet_model, UNetModel):
        return
    for module in unet_model.modules():
        if isinstance(module, BasicTransformerBlock):
            module.checkpoint = False
        if isinstance(module, ResBlock):
            module.use_checkpoint = False
    unet_model.convert_to_fp16()
    return compile(unet_model, backend="nexfort", options=options)


@torch.autocast("cuda", enabled=False)
def onediff_nexfort_unet_ldm_forward(
    orig_func, self, x, timesteps=None, context=None, y=None, **kwargs
):
    assert (y is not None) == (
        self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"
    hs = []
    t_emb = timestep_embedding(timesteps, self.model_channels).half()
    emb = self.time_embed(t_emb)
    x = x.half()
    context = context.half()
    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)
    h = x
    for module in self.input_blocks:
        h = module(h, emb, context)
        hs.append(h)
    h = self.middle_block(h, emb, context)
    for module in self.output_blocks:
        h = torch.cat([h, hs.pop()], dim=1)
        h = module(h, emb, context)
    if self.predict_codebook_ids:
        return self.id_predictor(h)
    else:
        return self.out(h)


@singleton_decorator
def create_cond_func():
    CondFunc(
        "ldm.modules.diffusionmodules.openaimodel.UNetModel.forward",
        onediff_nexfort_unet_ldm_forward,
        lambda orig_func, *args, **kwargs: onediff_shared.onediff_enabled,
    )


# def init_cond_func():
