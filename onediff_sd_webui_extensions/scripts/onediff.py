"""oneflow_compiled UNetModel"""


from pathlib import Path

import gradio as gr
import modules.scripts as scripts
import modules.sd_models as sd_models
import modules.shared as shared
import onediff_controlnet
import onediff_shared
from compile import (
    get_compiled_graph,
    get_onediff_backend,
    OneDiffBackend,
    SD21CompileCtx,
    VaeCompileCtx,
)
from compile.nexfort.utils import add_nexfort_optimizer
from modules import script_callbacks
from modules.processing import process_images
from modules.ui_common import create_refresh_button

from onediff.utils import logger, parse_boolean_from_env
from onediff_hijack import do_hijack as onediff_do_hijack
from onediff_lora import HijackLoraActivate

# from onediff.optimization.quant_optimizer import varify_can_use_quantization
from onediff_utils import (
    check_structure_change,
    get_all_compiler_caches,
    hints_message,
    load_graph,
    onediff_enabled_decorator,
    onediff_gc,
    refresh_all_compiler_caches,
    save_graph,
    varify_can_use_quantization,
)


class UnetCompileCtx(object):
    """The unet model is stored in a global variable.
    The global variables need to be replaced with compiled_unet before process_images is run,
    and then the original model restored so that subsequent reasoning with onediff disabled meets expectations.
    """

    def __init__(self, enabled):
        self.enabled = enabled

    def __enter__(self):
        if not self.enabled:
            return
        self._original_model = shared.sd_model.model.diffusion_model
        shared.sd_model.model.diffusion_model = (
            onediff_shared.current_unet_graph.graph_module
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        shared.sd_model.model.diffusion_model = self._original_model


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
        return True

    @onediff_enabled_decorator
    @onediff_controlnet.onediff_controlnet_decorator
    def run(
        self,
        p,
        quantization=False,
        compiler_cache=None,
        saved_cache_name="",
        always_recompile=False,
        backend=None,
    ):
        # restore checkpoint_info from refiner to base model if necessary
        if (
            sd_models.checkpoint_aliases.get(
                p.override_settings.get("sd_model_checkpoint")
            )
            is None
        ):
            p.override_settings.pop("sd_model_checkpoint", None)
            sd_models.reload_model_weights()
            onediff_gc()

        backend = backend or get_onediff_backend()
        current_checkpoint_name = shared.sd_model.sd_checkpoint_info.name
        ckpt_changed = (
            shared.sd_model.sd_checkpoint_info.name
            != onediff_shared.current_unet_graph.name
        )
        structure_changed = check_structure_change(
            onediff_shared.previous_unet_type, shared.sd_model
        )
        quantization_changed = (
            quantization != onediff_shared.current_unet_graph.quantized
        )
        controlnet_enabled_status_changed = (
            onediff_shared.controlnet_enabled != onediff_shared.previous_is_controlnet
        )
        need_recompile = (
            (
                quantization and ckpt_changed
            )  # always recompile when switching ckpt with 'int8 speed model' enabled
            or structure_changed  # always recompile when switching model to another structure
            or quantization_changed  # always recompile when switching model from non-quantized to quantized (and vice versa)
            or controlnet_enabled_status_changed
            or always_recompile
        )
        if need_recompile:
            if not onediff_shared.controlnet_enabled:
                onediff_shared.current_unet_graph = get_compiled_graph(
                    shared.sd_model,
                    quantization=quantization,
                    backend=backend,
                )
                load_graph(onediff_shared.current_unet_graph, compiler_cache)
        else:
            logger.info(
                f"Model {current_checkpoint_name} has same sd type of graph type {onediff_shared.previous_unet_type}, skip compile"
            )

        with UnetCompileCtx(not onediff_shared.controlnet_enabled), VaeCompileCtx(
            backend=backend
        ), SD21CompileCtx(), HijackLoraActivate():
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
    shared.opts.add_option(
        "onediff_compiler_backend",
        shared.OptionInfo(
            "oneflow",
            "Backend for onediff compiler (if you switch backend, you need to restart webui service)",
            gr.Radio,
            {"choices": [OneDiffBackend.ONEFLOW, OneDiffBackend.NEXFORT]},
            section=section,
        ),
    )


def cfg_denoisers_callback(params):
    pass


script_callbacks.on_ui_settings(on_ui_settings)
# script_callbacks.on_cfg_denoiser(cfg_denoisers_callback)
onediff_do_hijack()


script_callbacks.on_list_optimizers(add_nexfort_optimizer)
