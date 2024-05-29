import os
import warnings
import zipfile
from pathlib import Path
from typing import Dict, Union

import gradio as gr
import modules.scripts as scripts
import modules.shared as shared
from compile import (
    SD21CompileCtx,
    VaeCompileCtx,
    get_compiled_graph,
    OneDiffCompiledGraph,
)
from modules import script_callbacks
from modules.processing import process_images
from modules.sd_models import select_checkpoint
from modules.ui_common import create_refresh_button
from onediff_hijack import do_hijack as onediff_do_hijack
from onediff_lora import HijackLoraActivate
from oneflow import __version__ as oneflow_version
from ui_utils import (
    all_compiler_caches_path,
    get_all_compiler_caches,
    hints_message,
    refresh_all_compiler_caches,
    check_structure_change_and_update,
    load_graph,
    save_graph,
)

from onediff import __version__ as onediff_version
from onediff.optimization.quant_optimizer import (
    quantize_model,
    varify_can_use_quantization,
)
from onediff.utils import logger, parse_boolean_from_env
import onediff_shared

"""oneflow_compiled UNetModel"""
# compiled_unet = {}
# compiled_unet = None
# is_unet_quantized = False
# compiled_ckpt_name = None


def generate_graph_path(ckpt_name: str, model_name: str) -> str:
    base_output_dir = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples
    save_ckpt_graphs_path = os.path.join(base_output_dir, "graphs", ckpt_name)
    os.makedirs(save_ckpt_graphs_path, exist_ok=True)

    file_name = f"{model_name}_graph_{onediff_version}_oneflow_{oneflow_version}"

    graph_file_path = os.path.join(save_ckpt_graphs_path, file_name)

    return graph_file_path


def get_calibrate_info(filename: str) -> Union[None, Dict]:
    calibration_path = Path(select_checkpoint().filename).parent / filename
    if not calibration_path.exists():
        return None

    logger.info(f"Got calibrate info at {str(calibration_path)}")
    calibrate_info = {}
    with open(calibration_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            items = line.split(" ")
            calibrate_info[items[0]] = [
                float(items[1]),
                int(items[2]),
                [float(x) for x in items[3].split(",")],
            ]
    return calibrate_info


class UnetCompileCtx(object):
    """The unet model is stored in a global variable.
    The global variables need to be replaced with compiled_unet before process_images is run,
    and then the original model restored so that subsequent reasoning with onediff disabled meets expectations.
    """

    def __init__(self, compiled_unet):
        self.compiled_unet = compiled_unet

    def __enter__(self):
        self._original_model = shared.sd_model.model.diffusion_model
        shared.sd_model.model.diffusion_model = self.compiled_unet

    def __exit__(self, exc_type, exc_val, exc_tb):
        shared.sd_model.model.diffusion_model = self._original_model
        return False


class Script(scripts.Script):
    def title(self):
        return "onediff_diffusion_model"

    def ui(self, is_img2img):
        with gr.Row():
            # TODO: set choices as Tuple[str, str] after the version of gradio specified webui upgrades
            compiler_cache = gr.Dropdown(
                label="Compiler caches (Beta)",
                choices=["None"] + get_all_compiler_caches(),
                value="None",
                elem_id="onediff_compiler_cache",
            )
            create_refresh_button(
                compiler_cache,
                refresh_all_compiler_caches,
                lambda: {"choices": ["None"] + get_all_compiler_caches()},
                "onediff_refresh_compiler_caches",
            )
            save_cache_name = gr.Textbox(label="Saved cache name (Beta)")
        with gr.Row():
            always_recompile = gr.components.Checkbox(
                label="always_recompile",
                visible=parse_boolean_from_env("ONEDIFF_DEBUG"),
            )
        gr.HTML(
            hints_message,
            elem_id="hintMessage",
            visible=not varify_can_use_quantization(),
        )
        is_quantized = gr.components.Checkbox(
            label="Model Quantization(int8) Speed Up",
            visible=varify_can_use_quantization(),
        )
        return [is_quantized, compiler_cache, save_cache_name, always_recompile]

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def run(
        self,
        p,
        quantization=False,
        compiler_cache=None,
        saved_cache_name="",
        always_recompile=False,
    ):

        current_checkpoint_name = shared.sd_model.sd_checkpoint_info.name
        ckpt_changed = (
            shared.sd_model.sd_checkpoint_info.name
            != onediff_shared.current_unet_graph.name
        )
        structure_changed = check_structure_change_and_update(
            onediff_shared.current_unet_type, shared.sd_model
        )
        quantization_changed = (
            quantization != onediff_shared.current_unet_graph.quantized
        )
        need_recompile = (
            (
                quantization and ckpt_changed
            )  # always recompile when switching ckpt with 'int8 speed model' enabled
            or structure_changed  # always recompile when switching model to another structure
            or quantization_changed  # always recompile when switching model from non-quantized to quantized (and vice versa)
            or always_recompile
        )
        if need_recompile:
            onediff_shared.current_unet_graph = get_compiled_graph(
                shared.sd_model, quantization
            )
            load_graph(onediff_shared.current_unet_graph, compiler_cache)
        else:
            logger.info(
                f"Model {current_checkpoint_name} has same sd type of graph type {onediff_shared.current_unet_type}, skip compile"
            )

        # register graph
        onediff_shared.graph_dict[shared.sd_model.sd_model_hash] = OneDiffCompiledGraph(
            shared.sd_model, graph_module=onediff_shared.current_unet_graph.graph_module
        )
        with UnetCompileCtx(
            onediff_shared.current_unet_graph.graph_module
        ), VaeCompileCtx(), SD21CompileCtx(), HijackLoraActivate():
            proc = process_images(p)
        save_graph(onediff_shared.current_unet_graph, saved_cache_name)
        return proc


def on_ui_settings():
    section = ("onediff", "OneDiff")
    shared.opts.add_option(
        "onediff_compiler_caches_path",
        shared.OptionInfo(
            str(Path(__file__).parent.parent / "compiler_caches"),
            "Directory for onediff compiler caches",
            section=section,
        ),
    )


def cfg_denoisers_callback(params):
    # print(f"current checkpoint: {shared.opts.sd_model_checkpoint}")
    # import ipdb; ipdb.set_trace()
    if "refiner" in shared.sd_model.sd_checkpoint_info.name:
        pass
        # import ipdb; ipdb.set_trace()
        # shared.sd_model.model.diffusion_model

    print(f"current checkpoint info: {shared.sd_model.sd_checkpoint_info.name}")
    # shared.sd_model.model.diffusion_model = compile_unet(
    #     shared.sd_model.model.diffusion_model
    # )

    # have to check if onediff enabled
    # print('onediff denoiser callback')


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_cfg_denoiser(cfg_denoisers_callback)
onediff_do_hijack()
