import oneflow as flow
from onediff.infer_compiler import oneflow_compile, register
from sd_webui_onediff_utils import (
    CrossAttentionOflow,
    GroupNorm32Oflow,
    timestep_embedding,
)
from sgm.modules.attention import CrossAttention
from sgm.modules.diffusionmodules.openaimodel import UNetModel
from sgm.modules.diffusionmodules.util import GroupNorm32

__all__ = ["compile_sgm_unet"]


# https://github.com/Stability-AI/generative-models/blob/059d8e9cd9c55aea1ef2ece39abf605efb8b7cc9/sgm/modules/diffusionmodules/openaimodel.py#L816
class UNetModelOflow(flow.nn.Module):
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels).half()
        emb = self.time_embed(t_emb)
        x = x.half()
        context = context.half() if context is not None else context
        y = y.half() if y is not None else y
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
        h = h.type(x.dtype)
        return self.out(h)


torch2oflow_class_map = {
    CrossAttention: CrossAttentionOflow,
    GroupNorm32: GroupNorm32Oflow,
    UNetModel: UNetModelOflow,
}
register(package_names=["sgm"], torch2oflow_class_map=torch2oflow_class_map)


def compile_sgm_unet(unet_model, *, use_graph=True, options={}):
    if isinstance(unet_model, UNetModel):
        return oneflow_compile(unet_model, use_graph=use_graph, options=options)
