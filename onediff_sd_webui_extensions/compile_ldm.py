import math
import oneflow as flow
from onediff.infer_compiler import oneflow_compile, register

import compiled_model
from ldm.modules.attention import BasicTransformerBlock, CrossAttention
from ldm.modules.diffusionmodules.openaimodel import ResBlock, UNetModel
from ldm.modules.diffusionmodules.util import GroupNorm32
from sd_webui_onediff_utils import CrossAttentionOflow, GroupNorm32Oflow

__all__ = ["compile_ldm_unet"]


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = flow.exp(
        -math.log(max_period) * flow.arange(start=0, end=half, dtype=flow.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = flow.cat([flow.cos(args), flow.sin(args)], dim=-1)
    if dim % 2:
        embedding = flow.cat([embedding, flow.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class UNetModelOflow(flow.nn.Module):
    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
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
            h = flow.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


torch2oflow_class_map = {
    CrossAttention: CrossAttentionOflow,
    GroupNorm32: GroupNorm32Oflow,
    UNetModel: UNetModelOflow,
}
register(package_names=["ldm"],  torch2oflow_class_map=torch2oflow_class_map)


def compile_ldm_unet(sd_model):
    unet_model = sd_model.model.diffusion_model
    if not isinstance(unet_model, UNetModel):
        return
    for module in unet_model.modules():
        if isinstance(module, BasicTransformerBlock):
            module.checkpoint = False
        if isinstance(module, ResBlock):
            module.use_checkpoint = False
    compiled = oneflow_compile(unet_model, use_graph=True)
    compiled_model.compiled_unet = compiled
