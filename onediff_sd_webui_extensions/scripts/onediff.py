import modules.scripts as scripts
from modules import script_callbacks
import modules.shared as shared
from modules.processing import process_images

import torch
import oneflow as flow
from oneflow import nn
from sgm.modules.attention import CrossAttention
from sgm.modules.diffusionmodules.util import GroupNorm32
from omegaconf import OmegaConf, ListConfig
from onediff.infer_compiler.transform.builtin_transform import torch2oflow
from onediff.infer_compiler import oneflow_compile, register

import compiled_model
from compile_ldm import compile_ldm_unet
from sd_webui_onediff_utils import CrossAttentionOflow, GroupNorm32Oflow, TimeEmbedModule


@torch2oflow.register
def _(mod, verbose=False) -> ListConfig:
    converted_list = [torch2oflow(item, verbose) for item in mod]
    return OmegaConf.create(converted_list)


# https://github.com/Stability-AI/generative-models/blob/e5963321482a091a78375f3aeb2c3867562c913f/sgm/modules/diffusionmodules/wrappers.py#L24
def forward_wrapper( self, x, t, c, **kwargs):
    x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
    with torch.autocast("cuda", enabled=False):
        with flow.autocast("cuda", enabled=False):
            return self.diffusion_model(
                x.half(),
                timesteps=t.half(),
                context=c.get("crossattn", None).half(),
                y=c.get("vector", None).half(),
                **kwargs,
            )


torch2oflow_class_map = {
    CrossAttention: CrossAttentionOflow,
    GroupNorm32: GroupNorm32Oflow,
}
register(package_names=["sgm"], torch2oflow_class_map=torch2oflow_class_map)


def compile(sd_model):
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


class Script(scripts.Script):
    def title(self):
        return "onediff_diffusion_model"

    def show(self, is_img2img):
        return not is_img2img

    def run(self, p):
        if compiled_model.compiled_unet is None:
            compile(shared.sd_model)
        compiled = compiled_model.compiled_unet
        original = shared.sd_model.model.diffusion_model
        from sgm.modules.diffusionmodules.wrappers import OpenAIWrapper
        orig_forward = OpenAIWrapper.forward
        if compiled is not None:
            shared.sd_model.model.diffusion_model = compiled
            setattr(OpenAIWrapper, "forward", forward_wrapper)
        proc = process_images(p)
        shared.sd_model.model.diffusion_model = original
        setattr(OpenAIWrapper, "forward", orig_forward)
        return proc


script_callbacks.on_model_loaded(compile)
