import os
import oneflow as flow
from onediff.infer_compiler import oneflow_compile, register
from onediff.infer_compiler.transform import proxy_class

from ldm.modules.attention import BasicTransformerBlock, CrossAttention, SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import ResBlock, UNetModel
from ldm.modules.diffusionmodules.util import GroupNorm32
from modules import shared
from sd_webui_onediff_utils import (
    CrossAttentionOflow,
    GroupNorm32Oflow,
    timestep_embedding,
)

__all__ = ["compile_ldm_unet"]


# https://github.com/Stability-AI/stablediffusion/blob/b4bdae9916f628461e1e4edbc62aafedebb9f7ed/ldm/modules/diffusionmodules/openaimodel.py#L775
class UNetModelOflow(proxy_class(UNetModel)):
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
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


class SpatialTransformerOflow(proxy_class(SpatialTransformer)):
    # https://github.com/Stability-AI/stablediffusion/blob/e1797ae248408ea47561eeb8755737f1e35784f2/ldm/modules/attention.py#L321
    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = x.flatten(2, 3).permute(0, 2, 1)
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = x.permute(0, 2, 1).reshape_as(x_in)
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


torch2oflow_class_map = {
    CrossAttention: CrossAttentionOflow,
    GroupNorm32: GroupNorm32Oflow,
    SpatialTransformer: SpatialTransformerOflow,
    UNetModel: UNetModelOflow,
}
register(package_names=["ldm"], torch2oflow_class_map=torch2oflow_class_map)


def compile_ldm_unet(unet_model, *, use_graph=True, options={}):
    if not isinstance(unet_model, UNetModel):
        return
    for module in unet_model.modules():
        if isinstance(module, BasicTransformerBlock):
            module.checkpoint = False
        if isinstance(module, ResBlock):
            module.use_checkpoint = False
    return oneflow_compile(unet_model, use_graph=use_graph, options=options)


class SD21CompileCtx(object):
    """to avoid results for NaN when the model is v2-1_768-ema-pruned"""

    _var_name = "ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_ACCUMULATION"

    def __enter__(self):
        self._original = os.getenv(self._var_name)
        if shared.opts.sd_model_checkpoint.startswith("v2-1"):
            os.environ[self._var_name] = "0"

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original is not None:
            os.environ[self._var_name] = self._original
        return False
