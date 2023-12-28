import modules.scripts as scripts
from modules import script_callbacks
import modules.shared as shared
from modules.processing import process_images

from onediff.infer_compiler.transform.builtin_transform import torch2oflow
from omegaconf import OmegaConf, ListConfig

from compile_ldm import compile_ldm_unet
from compile_sgm import compile_sgm_unet
from compile_vae import VaeCompileCtx


@torch2oflow.register
def _(mod, verbose=False) -> ListConfig:
    converted_list = [torch2oflow(item, verbose) for item in mod]
    return OmegaConf.create(converted_list)


"""oneflow_compiled UNetModel"""
compiled_unet = None


def compile(sd_model):
    from ldm.modules.diffusionmodules.openaimodel import UNetModel as UNetModelLDM
    from sgm.modules.diffusionmodules.openaimodel import UNetModel as UNetModelSGM

    unet_model = sd_model.model.diffusion_model
    global compiled_unet
    if isinstance(unet_model, UNetModelLDM):
        compiled_unet = compile_ldm_unet(sd_model)
    elif isinstance(unet_model, UNetModelSGM):
        compiled_unet = compile_sgm_unet(sd_model)


def supplement_sys_path():
    """add package path to sys.path to avoid mock error"""
    import ldm, sgm, sys

    sys_paths = set(sys.path)
    new_paths = [sgm.__path__[0][:-4], ldm.__path__[0][:-4]]
    for path in new_paths:
        if path not in sys_paths:
            sys.path.append(path)


class UnetCompileCtx(object):
    def __enter__(self):
        self._original_model = shared.sd_model.model.diffusion_model
        global compiled_unet
        if compiled_unet is None:
            compiled_unet = compile(shared.sd_model)
        shared.sd_model.model.diffusion_model = compiled_unet

    def __exit__(self, exc_type, exc_val, exc_tb):
        shared.sd_model.model.diffusion_model = self._original_model
        return False


class Script(scripts.Script):
    def title(self):
        return "onediff_diffusion_model"

    def show(self, is_img2img):
        return not is_img2img

    def run(self, p):
        with UnetCompileCtx(), VaeCompileCtx():
            supplement_sys_path()
            proc = process_images(p)
        return proc


script_callbacks.on_model_loaded(compile)
