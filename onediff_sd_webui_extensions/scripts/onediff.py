import os
import warnings
import gradio as gr
import modules.scripts as scripts
import modules.shared as shared
from modules.processing import process_images

from compile_ldm import compile_ldm_unet, SD21CompileCtx
from compile_sgm import compile_sgm_unet
from compile_vae import VaeCompileCtx

from onediff.optimization.quant_optimizer import (
    quantize_model,
    varify_can_use_quantization,
)

"""oneflow_compiled UNetModel"""
compiled_unet = None
compiled_ckpt_name = None


def generate_graph_path(ckpt_name: str, model_name: str) -> str:
    base_output_dir = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples
    save_ckpt_graphs_path = os.path.join(base_output_dir, "graphs", ckpt_name)
    os.makedirs(save_ckpt_graphs_path, exist_ok=True)
    graph_file_path = os.path.join(save_ckpt_graphs_path, f"{model_name}.graph")
    return graph_file_path


def is_compiled(ckpt_name):
    global compiled_unet, compiled_ckpt_name

    return compiled_unet is not None and compiled_ckpt_name == ckpt_name


def compile_unet(
    unet_model, quantization=False, *, use_graph=True, options={},
):
    from ldm.modules.diffusionmodules.openaimodel import UNetModel as UNetModelLDM
    from sgm.modules.diffusionmodules.openaimodel import UNetModel as UNetModelSGM

    if quantization:
        unet_model = quantize_model(unet_model, inplace=False)

    if isinstance(unet_model, UNetModelLDM):
        return compile_ldm_unet(unet_model, use_graph=use_graph, options=options)
    elif isinstance(unet_model, UNetModelSGM):
        return compile_sgm_unet(unet_model, use_graph=use_graph, options=options)
    else:
        warnings.warn(
            f"Unsupported model type: {type(unet_model)} for compilation , skip",
            RuntimeWarning,
        )
        return unet_model


class UnetCompileCtx(object):
    """The unet model is stored in a global variable.
    The global variables need to be replaced with compiled_unet before process_images is run,
    and then the original model restored so that subsequent reasoning with onediff disabled meets expectations.
    """

    def __enter__(self):
        self._original_model = shared.sd_model.model.diffusion_model
        global compiled_unet
        shared.sd_model.model.diffusion_model = compiled_unet

    def __exit__(self, exc_type, exc_val, exc_tb):
        shared.sd_model.model.diffusion_model = self._original_model
        return False

class HijackLoraActivate(object):
    def hijacked_activate(activate_func):
        import torch
        from onediff.infer_compiler.with_oneflow_compile import DeployableModule
        import networks
        def activate(self, p, params_list):
            # When switch from OneDiff to PyTorch, modules with LoRA need to reload LoRA
            # since `network_apply_weights` in forward is not called
            if not hasattr(self, "switch_from_onediff"):
                self.switch_from_onediff = False
            if self.switch_from_onediff and not isinstance(p.sd_model.model.diffusion_model, DeployableModule):
                for name, sub_module in p.sd_model.model.diffusion_model.named_modules():
                    if hasattr(sub_module, "network_current_names"):
                        setattr(sub_module, "network_current_names", ())
                self.switch_from_onediff = False
            activate_func(self, p, params_list)
            if isinstance(p.sd_model.model.diffusion_model, DeployableModule):
                self.switch_from_onediff = True
                onediff_sd_model: DeployableModule = p.sd_model.model.diffusion_model
                for name, sub_module in onediff_sd_model.named_modules():
                    if not isinstance(
                        sub_module,
                        (
                            torch.nn.Linear,
                            torch.nn.Conv2d,
                            torch.nn.GroupNorm,
                            torch.nn.LayerNorm,
                        ),
                    ):
                        continue
                    networks.network_apply_weights(sub_module)
        return activate

    def hijacked_deactivate(deactivate_func):
        import torch
        from onediff.infer_compiler.with_oneflow_compile import DeployableModule
        import networks
        def deactivate(self, p):
            deactivate_func(self, p)
            if isinstance(p.sd_model.model.diffusion_model, DeployableModule):
                onediff_sd_model: DeployableModule = p.sd_model.model.diffusion_model
                for name, sub_module in onediff_sd_model.named_modules():
                    if not isinstance(
                        sub_module,
                        (
                            torch.nn.Linear,
                            torch.nn.Conv2d,
                            torch.nn.GroupNorm,
                            torch.nn.LayerNorm,
                        ),
                    ):
                        continue
                    with torch.no_grad():
                        networks.network_restore_weights_from_backup(sub_module)
        return deactivate

    def __init__(self):
        self.original_activate_func = None
        self.original_deactivate_func = None


    def __enter__(self):
        from modules import extra_networks
        if 'lora' in extra_networks.extra_network_registry:
            # Hijick
            cls_extra_network_lora = type(extra_networks.extra_network_registry['lora'])
            self.original_activate_func = cls_extra_network_lora.activate
            self.original_deactivate_func = cls_extra_network_lora.deactivate
            cls_extra_network_lora.activate = HijackLoraActivate.hijacked_activate(cls_extra_network_lora.activate)
            cls_extra_network_lora.deactivate = HijackLoraActivate.hijacked_deactivate(cls_extra_network_lora.deactivate)

    
    def __exit__(self, exc_type, exc_val, exc_tb):
        from modules import extra_networks
        if 'lora' in extra_networks.extra_network_registry and self.original_activate_func is not None and self.original_deactivate_func is not None:
            cls_extra_network_lora = type(extra_networks.extra_network_registry['lora'])
            cls_extra_network_lora.activate = self.original_activate_func
            cls_extra_network_lora.deactivate = self.original_deactivate_func
            self.original_activate_func = None
            self.original_deactivate_func = None




class Script(scripts.Script):
    def title(self):
        return "onediff_diffusion_model"

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        if not varify_can_use_quantization():
            ret = gr.HTML(
                """
                    <div style="padding: 20px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f9f9f9;">
                        <div style="font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #31708f;">
                            Hints Message
                        </div>
                        <div style="padding: 10px; border: 1px solid #31708f; border-radius: 5px; background-color: #f9f9f9;">
                            Hints: Enterprise function is not supported on your system.
                        </div>
                        <p style="margin-top: 15px;">
                            If you need Enterprise Level Support for your system or business, please send an email to 
                            <a href="mailto:business@siliconflow.com" style="color: #31708f; text-decoration: none;">business@siliconflow.com</a>.
                            <br>
                            Tell us about your use case, deployment scale, and requirements.
                        </p>
                        <p>
                            <strong>GitHub Issue:</strong>
                            <a href="https://github.com/siliconflow/onediff/issues" style="color: #31708f; text-decoration: none;">https://github.com/siliconflow/onediff/issues</a>
                        </p>
                    </div>
                    """
            )

        else:
            ret = gr.components.Checkbox(label="Model Quantization(int8) Speed Up")
        return [ret]

    def show(self, is_img2img):
        return not is_img2img

    def run(self, p, quantization=False):
        global compiled_unet, compiled_ckpt_name
        current_checkpoint = shared.opts.sd_model_checkpoint
        original_diffusion_model = shared.sd_model.model.diffusion_model

        ckpt_name = (
            current_checkpoint + "_quantized" if quantization else current_checkpoint
        )

        if not is_compiled(ckpt_name):
            compiled_unet = compile_unet(
                original_diffusion_model, quantization=quantization,
            )
            compiled_ckpt_name = ckpt_name

        from modules import extra_networks
        if 'lora' in extra_networks.extra_network_registry:
            # Hijack
            cls_extra_network_lora = type(extra_networks.extra_network_registry['lora'])
            cls_extra_network_lora.activate = HijackLoraActivate.hijacked_activate(cls_extra_network_lora.activate)
            cls_extra_network_lora.deactivate = HijackLoraActivate.hijacked_deactivate(cls_extra_network_lora.deactivate)

        with UnetCompileCtx(), VaeCompileCtx(), SD21CompileCtx():
            proc = process_images(p)
        return proc
