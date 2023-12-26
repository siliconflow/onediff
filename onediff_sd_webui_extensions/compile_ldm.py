from onediff.infer_compiler import oneflow_compile, register

import compiled_model
from ldm.modules.attention import BasicTransformerBlock, CrossAttention
from ldm.modules.diffusionmodules.openaimodel import ResBlock, UNetModel
from ldm.modules.diffusionmodules.util import GroupNorm32
from sd_webui_onediff_utils import CrossAttentionOflow, GroupNorm32Oflow, TimeEmbedModule

torch2oflow_class_map = {
    CrossAttention: CrossAttentionOflow,
    GroupNorm32: GroupNorm32Oflow,
}

import ldm
register(package_names=[ldm.__path__[0][:-4]],  torch2oflow_class_map=torch2oflow_class_map)

__all__ = ["compile_ldm_unet"]


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
    # add ldm package path to sys.path to avoid mock errors
    import ldm, sys
    sys.path.append(ldm.__path__[0][:-4])
    of_model = compiled._deployable_module_model.oneflow_module
    time_embed_wrapper = TimeEmbedModule(of_model.time_embed)
    setattr(of_model, "time_embed", time_embed_wrapper)
    compiled_model.compiled_unet = compiled
