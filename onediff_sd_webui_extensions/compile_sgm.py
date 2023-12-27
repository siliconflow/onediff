import compiled_model
from onediff.infer_compiler.transform.builtin_transform import torch2oflow
from onediff.infer_compiler import oneflow_compile, register
from sd_webui_onediff_utils import CrossAttentionOflow, GroupNorm32Oflow, TimeEmbedModule
from sgm.modules.attention import CrossAttention
from sgm.modules.diffusionmodules.util import GroupNorm32

__all__ = ["compile_sgm_unet"]


torch2oflow_class_map = {
    CrossAttention: CrossAttentionOflow,
    GroupNorm32: GroupNorm32Oflow,
}
register(package_names=["sgm"], torch2oflow_class_map=torch2oflow_class_map)


def compile_sgm_unet(sd_model):
    unet_model = sd_model.model.diffusion_model
    full_name = f"{unet_model.__module__}.{unet_model.__class__.__name__}"
    if not full_name.endswith(".UNetModel"):
        return
    if full_name.startswith("ldm"):
        compile_ldm_unet(sd_model)
    compiled = oneflow_compile(sd_model.model.diffusion_model, use_graph=True)
    # add sgm package path to sys.path to avoid mock error
    import sgm, sys
    sys.path.append(sgm.__path__[0][:-4])
    time_embed_wrapper = TimeEmbedModule(compiled._deployable_module_model.oneflow_module.time_embed)
    # https://github.com/Stability-AI/generative-models/blob/e5963321482a091a78375f3aeb2c3867562c913f/sgm/modules/diffusionmodules/openaimodel.py#L984
    setattr(compiled._deployable_module_model.oneflow_module, "time_embed", time_embed_wrapper)
    compiled_model.compiled_unet = compiled
