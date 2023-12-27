import modules.scripts as scripts
from modules import script_callbacks
import modules.shared as shared
from modules.processing import process_images

from onediff.infer_compiler.transform.builtin_transform import torch2oflow
from omegaconf import OmegaConf, ListConfig

from compile_sgm import compile_sgm_unet
from compile_ldm import compile_ldm_unet


@torch2oflow.register
def _(mod, verbose=False) -> ListConfig:
    converted_list = [torch2oflow(item, verbose) for item in mod]
    return OmegaConf.create(converted_list)


"""oneflow_compiled UNetModel"""
compiled_unet = None

def generate_graph_path(ckpt_name, model_name):
    from pathlib import Path
    output_dir = Path("/home/fengwen/sd_webui/stable-diffusion-webui/outputs")/ "graphs"
    output_dir.mkdir(exist_ok=True)
    graph_path = output_dir / ckpt_name / f"{model_name}.graph"
    return graph_path


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


class Script(scripts.Script):
    def title(self):
        return "onediff_diffusion_model"

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        return [gr.components.Checkbox(label="Model Quantization(int8) Speed Up")]

    def show(self, is_img2img):
        return not is_img2img


    def run(self, p):
        global compiled_unet
        if compiled_unet is None:
            compiled_unet = compile(shared.sd_model)
        original = shared.sd_model.model.diffusion_model
        shared.sd_model.model.diffusion_model = compiled_unet
        supplement_sys_path()

        proc = process_images(p)
        shared.sd_model.model.diffusion_model = original
        return proc


# script_callbacks.on_model_loaded(compile)
