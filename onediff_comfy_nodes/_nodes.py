from functools import partial
from onediff.infer_compiler.with_oneflow_compile import oneflow_compile
from ._config import _USE_UNET_INT8

import os
import re
import time
import copy
import warnings
from pathlib import Path

import torch
import comfy
import folder_paths
from comfy import model_management
from comfy.cli_args import args
from folder_paths import get_input_directory

from .utils import (
    OneFlowSpeedUpModelPatcher,
    OneFlowDeepCacheSpeedUpModelPatcher,
    save_graph,
    load_graph,
    OUTPUT_FOLDER,
)

from .modules.hijack_model_management import model_management_hijacker

model_management_hijacker.hijack()  # add flow.cuda.empty_cache()


__all__ = [
    "ModelSpeedup",
    "ModelGraphLoader",
    "ModelGraphSaver",
    "UNETLoaderInt8",
    "VaeSpeedup",
    "VaeGraphLoader",
    "VaeGraphSaver",
    "SVDSpeedup",
    "ModuleDeepCacheSpeedup",
]

if not args.dont_upcast_attention:
    os.environ["ONEFLOW_KERENL_FMHA_ENABLE_TRT_FLASH_ATTN_IMPL"] = "0"


class ModelSpeedup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "static_mode": (["enable", "disable"],),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "speedup"
    CATEGORY = "OneDiff"

    def speedup(self, model, static_mode):
        from onediff.infer_compiler import oneflow_compile

        use_graph = static_mode == "enable"

        offload_device = model_management.unet_offload_device()
        oneflow_model = OneFlowSpeedUpModelPatcher(
            model.model,
            load_device=model_management.get_torch_device(),
            offload_device=offload_device,
            use_graph=use_graph,
        )

        return (oneflow_model,)


class UNETLoaderInt8:
    @classmethod
    def INPUT_TYPES(cls):
        paths = []
        for search_path in folder_paths.get_folder_paths("unet_int8"):
            if os.path.exists(search_path):
                for root, subdir, files in os.walk(search_path, followlinks=True):
                    if "calibrate_info.txt" in files:
                        paths.append(os.path.relpath(root, start=search_path))

        return {
            "required": {
                "model_path": (paths,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet_int8"

    CATEGORY = "OneDiff"

    def load_unet_int8(self, model_path):
        from .utils.diffusers_quant_utils import replace_module_with_quantizable_module

        for search_path in folder_paths.get_folder_paths("unet_int8"):
            if os.path.exists(search_path):
                path = os.path.join(search_path, model_path)
                if os.path.exists(path):
                    model_path = path
                    break

        unet_sd_path = os.path.join(model_path, "unet_int8.safetensors")
        calibrate_info_path = os.path.join(model_path, "calibrate_info.txt")

        model = comfy.sd.load_unet(unet_sd_path)
        replace_module_with_quantizable_module(
            model.model.diffusion_model, calibrate_info_path
        )
        return (model,)


class ModelGraphLoader:
    @classmethod
    def INPUT_TYPES(s):
        unet_folder = os.path.join(OUTPUT_FOLDER, "unet")
        graph_files = [
            f
            for f in os.listdir(unet_folder)
            if os.path.isfile(os.path.join(unet_folder, f)) and f.endswith(".graph")
        ]
        return {
            "required": {
                "model": ("MODEL",),
                "graph": (sorted(graph_files),),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_graph"
    CATEGORY = "OneDiff"

    def load_graph(self, model, graph):
        from onediff.infer_compiler.with_oneflow_compile import DeployableModule

        diffusion_model = model.model.diffusion_model

        load_graph(diffusion_model, graph, "cuda", subfolder="unet")
        return (model,)


class ModelGraphSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "unet"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_graph"
    CATEGORY = "OneDiff"
    OUTPUT_NODE = True

    def save_graph(self, samples, model, filename_prefix):
        diffusion_model = model.model.diffusion_model
        save_graph(diffusion_model, filename_prefix, "cuda", subfolder="unet")
        return {}


class SVDSpeedup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "static_mode": (["enable", "disable"],),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "speedup"
    CATEGORY = "OneDiff"

    def speedup(self, model, static_mode):
        from onediff.infer_compiler import oneflow_compile

        use_graph = static_mode == "enable"

        new_model = copy.deepcopy(model)
        new_model.model.diffusion_model = oneflow_compile(
            new_model.model.diffusion_model, use_graph=use_graph
        )

        return (new_model,)


class VaeSpeedup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "static_mode": (["enable", "disable"],),
            },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "speedup"
    CATEGORY = "OneDiff"

    def speedup(self, vae, static_mode):
        from onediff.infer_compiler import oneflow_compile

        use_graph = static_mode == "enable"

        new_vae = copy.deepcopy(
            vae
        )  # Loading/offloading will not cause an increase in VRAM.

        new_vae.first_stage_model = oneflow_compile(
            new_vae.first_stage_model, use_graph=use_graph
        )

        return (new_vae,)


class VaeGraphLoader:
    @classmethod
    def INPUT_TYPES(s):
        vae_folder = os.path.join(OUTPUT_FOLDER, "vae")
        graph_files = [
            f
            for f in os.listdir(vae_folder)
            if os.path.isfile(os.path.join(vae_folder, f)) and f.endswith(".graph")
        ]
        return {
            "required": {
                "vae": ("VAE",),
                "graph": (sorted(graph_files),),
            },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_graph"
    CATEGORY = "OneDiff"

    def load_graph(self, vae, graph):
        from onediff.infer_compiler.with_oneflow_compile import DeployableModule

        vae_model = vae.first_stage_model
        device = model_management.vae_offload_device()
        load_graph(vae_model, graph, device, subfolder="vae")
        return (vae,)


class VaeGraphSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "vae": ("VAE",),
                "filename_prefix": ("STRING", {"default": "vae"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_graph"
    CATEGORY = "OneDiff"
    OUTPUT_NODE = True

    def save_graph(self, images, vae, filename_prefix):
        from onediff.infer_compiler.with_oneflow_compile import DeployableModule

        vae_model = vae.first_stage_model
        vae_device = model_management.vae_offload_device()
        save_graph(vae_model, filename_prefix, vae_device, subfolder="vae")

        return {}


class ControlNetSpeedup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "static_mode": (["enable", "disable"],),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    RETURN_NAMES = ("control_net",)
    FUNCTION = "apply_controlnet"

    CATEGORY = "OneDiff"

    def apply_controlnet(self, control_net, static_mode):
        if static_mode == "enable":
            from comfy.controlnet import ControlNet, ControlLora
            from .modules.onediff_controlnet import (
                OneDiffControlNet,
                OneDiffControlLora,
            )

            if isinstance(control_net, ControlLora):
                control_net = OneDiffControlLora.from_controllora(control_net)
                return (control_net,)
            elif isinstance(control_net, ControlNet):
                control_net = OneDiffControlNet.from_controlnet(control_net)
                return (control_net,)
            else:
                raise TypeError(
                    f"control_net must be ControlNet or ControlLora, got {type(control_net)}"
                )
        else:
            return (control_net,)


class ControlNetGraphLoader:
    @classmethod
    def INPUT_TYPES(s):
        folder = os.path.join(OUTPUT_FOLDER, "control_net")
        graph_files = [
            f
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and f.endswith(".graph")
        ]
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "graph": (sorted(graph_files),),
            },
        }

    RETURN_TYPES = ("CONTROL_NET",)
    RETURN_NAMES = ("control_net",)
    FUNCTION = "load_graph"
    CATEGORY = "OneDiff"

    def load_graph(self, control_net, graph):
        from .modules.onediff_controlnet import HijackControlLora

        device = model_management.get_torch_device()

        lazy_load_hook = partial(
            load_graph, graph_filename=graph, device=device, subfolder="control_net"
        )
        setattr(HijackControlLora, "lazy_load_hook", lazy_load_hook)

        return (control_net,)


class ControlNetGraphSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "control_net": ("CONTROL_NET",),
                "filename_prefix": ("STRING", {"default": "control_net"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_graph"
    CATEGORY = "OneDiff"
    OUTPUT_NODE = True

    def save_graph(self, samples, control_net, filename_prefix):
        from .modules.onediff_controlnet import HijackControlLora

        # Unable to directly fetch the controlnet model from comfyui.
        model = HijackControlLora.oneflow_model
        device = model_management.get_torch_device()

        save_graph(model, filename_prefix, device, subfolder="control_net")
        return {}


class Quant8Model:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "output_dir": ("STRING", {"default": "int8"}),
                "conv": (["enable", "disable"],),
                "linear": (["enable", "disable"],),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "quantize_model"
    CATEGORY = "OneDiff"
    OUTPUT_NODE = True

    def quantize_model(self, model, output_dir, conv, linear):
        from .utils import quantize_and_save_model

        diffusion_model = model.model.diffusion_model
        output_dir = os.path.join(folder_paths.models_dir, "unet_int8", output_dir)
        is_quantize_conv = conv == "enable"
        is_quantize_linear = linear == "enable"
        quantize_and_save_model(
            diffusion_model,
            output_dir,
            quantize_conv=is_quantize_conv,
            quantize_linear=is_quantize_linear,
            verbose=False,
        )
        return {}


class ModuleDeepCacheSpeedup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "static_mode": (["enable", "disable"],),
                "cache_interval": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "cache_layer_id": (
                    "INT",
                    {"default": 0, "min": 0, "max": 12, "step": 1, "display": "number"},
                ),
                "cache_block_id": (
                    "INT",
                    {"default": 1, "min": 0, "max": 12, "step": 1, "display": "number"},
                ),
                "start_step": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "end_step": (
                    "INT",
                    {
                        "default": 1000,
                        "min": 0,
                        "max": 1000,
                        "step": 0.1,
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "deep_cache_convert"
    CATEGORY = "OneDiff"

    def deep_cache_convert(
        self,
        model,
        static_mode,
        cache_interval,
        cache_layer_id,
        cache_block_id,
        start_step,
        end_step,
    ):
        use_graph = static_mode == "enable"

        offload_device = model_management.unet_offload_device()
        oneflow_model = OneFlowDeepCacheSpeedUpModelPatcher(
            model.model,
            load_device=model_management.get_torch_device(),
            offload_device=offload_device,
            cache_layer_id=cache_layer_id,
            cache_block_id=cache_block_id,
            use_graph=use_graph,
        )

        current_t = -1
        current_step = -1
        cache_h = None

        def apply_model(model_function, kwargs):
            nonlocal current_t, current_step, cache_h

            xa = kwargs["input"]
            t = kwargs["timestep"]
            c_concat = kwargs["c"].get("c_concat", None)
            c_crossattn = kwargs["c"].get("c_crossattn", None)
            y = kwargs["c"].get("y", None)
            control = kwargs["c"].get("control", None)
            transformer_options = kwargs["c"].get("transformer_options", None)

            # https://github.com/comfyanonymous/ComfyUI/blob/629e4c552cc30a75d2756cbff8095640af3af163/comfy/model_base.py#L51-L69
            sigma = t
            xc = oneflow_model.model.model_sampling.calculate_input(sigma, xa)
            if c_concat is not None:
                xc = torch.cat([xc] + [c_concat], dim=1)

            context = c_crossattn
            dtype = oneflow_model.model.get_dtype()
            xc = xc.to(dtype)
            t = oneflow_model.model.model_sampling.timestep(t).float()
            context = context.to(dtype)
            extra_conds = {}
            for o in kwargs:
                extra = kwargs[o]
                if hasattr(extra, "to"):
                    extra = extra.to(dtype)
                extra_conds[o] = extra

            x = xc
            timesteps = t
            y = None if y is None else y.to(dtype)
            transformer_options["original_shape"] = list(x.shape)
            transformer_options["current_index"] = 0
            transformer_patches = transformer_options.get("patches", {})
            """
            Apply the model to an input batch.
            :param x: an [N x C x ...] Tensor of inputs.
            :param timesteps: a 1-D batch of timesteps.
            :param context: conditioning plugged in via crossattn
            :param y: an [N] Tensor of labels, if class-conditional.
            :return: an [N x C x ...] Tensor of outputs.
            """

            # reference https://gist.github.com/laksjdjf/435c512bc19636e9c9af4ee7bea9eb86
            if t[0].item() > current_t:
                current_step = -1

            current_t = t[0].item()
            apply = 1000 - end_step <= current_t <= 1000 - start_step  # t is 999->0

            if apply:
                current_step += 1
            else:
                current_step = -1
            current_t = t[0].item()

            is_slow_step = current_step % cache_interval == 0 and apply

            model_output = None
            if is_slow_step:
                cache_h = None
                model_output, cache_h = oneflow_model.deep_cache_unet(
                    x,
                    timesteps,
                    context,
                    y,
                    control,
                    transformer_options,
                    **extra_conds,
                )
            else:
                model_output, cache_h = oneflow_model.fast_deep_cache_unet(
                    x,
                    cache_h,
                    timesteps,
                    context,
                    y,
                    control,
                    transformer_options,
                    **extra_conds,
                )

            return oneflow_model.model.model_sampling.calculate_denoised(
                sigma, model_output, xa
            )

        oneflow_model.set_model_unet_function_wrapper(apply_model)
        return (oneflow_model,)


def generate_graph_path(ckpt_name, model):
    input_dir = get_input_directory()
    input_dir = Path(input_dir)
    graph_dir = input_dir / "graphs" / ckpt_name
    graph_file_path = graph_dir / (type(model).__name__ + ".graph")
    return graph_file_path


from nodes import CheckpointLoaderSimple


class OneDiffCheckpointLoaderSimple(CheckpointLoaderSimple):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_speedup": (["disable", "enable"],),
            }
        }

    CATEGORY = "OneDiff"

    def load_checkpoint(
        self, ckpt_name, output_vae=True, output_clip=True, vae_speedup="disable"
    ):
        model, clip, vae = super().load_checkpoint(ckpt_name, output_vae, output_clip)
        offload_device = model_management.unet_offload_device()

        diffusion_model = model.model.diffusion_model
        file_path = generate_graph_path(ckpt_name, diffusion_model)
        print(f" OneDiffCheckpointLoaderSimple load_checkpoint file_path {file_path}")

        oneflow_model = OneFlowSpeedUpModelPatcher(
            model.model,
            load_device=model_management.get_torch_device(),
            offload_device=offload_device,
            use_graph=True,
            graph_path=file_path,
            graph_device=model_management.get_torch_device(),
        )

        if vae_speedup == "enable":
            file_path = generate_graph_path(ckpt_name, vae.first_stage_model)
            vae.first_stage_model = oneflow_compile(
                vae.first_stage_model,
                use_graph=True,
                options={
                    "graph_file": file_path,
                    "graph_file_device": model_management.get_torch_device(),
                },
            )
        return oneflow_model, clip, vae