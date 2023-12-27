import modules.scripts as scripts
from modules import script_callbacks
import modules.shared as shared
from modules.processing import process_images

import torch
import oneflow as flow
from onediff.infer_compiler.transform.builtin_transform import torch2oflow
from omegaconf import OmegaConf, ListConfig

import compiled_model
from compile_sgm import compile_sgm_unet
from compile_ldm import compile_ldm_unet


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


def compile(sd_model):
    from ldm.modules.diffusionmodules.openaimodel import UNetModel as UNetModelLDM
    from sgm.modules.diffusionmodules.openaimodel import UNetModel as UNetModelSGM
    unet_model = sd_model.model.diffusion_model
    if isinstance(unet_model, UNetModelLDM):
        compile_ldm_unet(sd_model)
    elif isinstance(unet_model, UNetModelSGM):
        compile_sgm_unet(sd_model)


class Script(scripts.Script):
    def title(self):
        return "onediff_diffusion_model"

    def show(self, is_img2img):
        return not is_img2img

    def run(self, p):
        if compiled_model.compiled_unet is None:
            # compile and save result to compiled_model.compiled_unet
            compile(shared.sd_model)
        compiled = compiled_model.compiled_unet
        original = shared.sd_model.model.diffusion_model
        # set OpenAIWrapper.forward for sgm unet
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
