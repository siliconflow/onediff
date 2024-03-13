from functools import partial
from onediff.infer_compiler.transform import torch2oflow
from onediff.infer_compiler.with_oneflow_compile import oneflow_compile
from ._config import _USE_UNET_INT8, ONEDIFF_QUANTIZED_OPTIMIZED_MODELS
from onediff.infer_compiler.utils import set_boolean_env_var
from onediff.optimization.quant_optimizer import quantize_model


import os
import re
import time
import copy
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import comfy
import folder_paths
from comfy import model_management
from comfy.cli_args import args
from .utils import (
    OneFlowSpeedUpModelPatcher,
    OneFlowDeepCacheSpeedUpModelPatcher,
    save_graph,
    load_graph,
    OUTPUT_FOLDER,
)
from .utils.model_patcher import state_dict_hook, get_mixed_speedup_class
from .utils.loader_sample_tools import compoile_unet, quantize_unet
from .utils.graph_path import generate_graph_path
from .modules.hijack_model_management import model_management_hijacker
from .modules.hijack_nodes import nodes_hijacker
from .utils.deep_cache_speedup import deep_cache_speedup
from .utils.onediff_load_utils import onediff_load_quant_checkpoint_advanced

model_management_hijacker.hijack()  # add flow.cuda.empty_cache()
nodes_hijacker.hijack()


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
    os.environ["ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_SCORE_ACCUMULATION_MAX_M"] = "0"


class ModelSpeedup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"model": ("MODEL",), "static_mode": (["enable", "disable"],),},
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "speedup"
    CATEGORY = "OneDiff"

    def speedup(self, model, static_mode):
        from onediff.infer_compiler import oneflow_compile

        use_graph = static_mode == "enable"

        offload_device = model_management.unet_offload_device()
        oneflow_model = get_mixed_speedup_class(model.__class__)(
            model,
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

        return {"required": {"model_path": (paths,),}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet_int8"

    CATEGORY = "OneDiff"

    def load_unet_int8(self, model_path):
        from .utils.onediff_quant_utils import replace_module_with_quantizable_module

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
            "required": {"model": ("MODEL",), "graph": (sorted(graph_files),),},
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
            "required": {"model": ("MODEL",), "static_mode": (["enable", "disable"],),},
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "speedup"
    CATEGORY = "OneDiff"

    def speedup(self, model, static_mode):
        from onediff.infer_compiler import oneflow_compile

        # To avoid overflow issues while maintaining performance,
        # refer to: https://github.com/siliconflow/onediff/blob/09a94df1c1a9c93ec8681e79d24bcb39ff6f227b/examples/image_to_video.py#L112
        set_boolean_env_var(
            "ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_SCORE_ACCUMULATION_MAX_M", False
        )

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
            "required": {"vae": ("VAE",), "static_mode": (["enable", "disable"],),},
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
            "required": {"vae": ("VAE",), "graph": (sorted(graph_files),),},
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
                    {"default": 1000, "min": 0, "max": 1000, "step": 0.1,},
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
        return deep_cache_speedup(
            model=model,
            use_graph=(static_mode == "enable"),
            cache_interval=cache_interval,
            cache_layer_id=cache_layer_id,
            cache_block_id=cache_block_id,
            start_step=start_step,
            end_step=end_step,
        )


import comfy_extras.nodes_video_model
from nodes import CheckpointLoaderSimple, ControlNetLoader
from comfy.controlnet import ControlLora, ControlNet

from .modules.onediff_controlnet import OneDiffControlLora


class OneDiffControlNetLoader(ControlNetLoader):
    CATEGORY = "OneDiff/Loaders"
    FUNCTION = "onediff_load_controlnet"

    def onediff_load_controlnet(self, control_net_name):
        controlnet = super().load_controlnet(control_net_name)[0]
        load_device = model_management.get_torch_device()

        def gen_compile_options(model):
            graph_file = generate_graph_path(control_net_name, model)
            return {
                "graph_file": graph_file,
                "graph_file_device": load_device,
            }

        if isinstance(controlnet, ControlLora):
            controlnet = OneDiffControlLora.from_controllora(
                controlnet, gen_compile_options=gen_compile_options
            )
            return (controlnet,)
        elif isinstance(controlnet, ControlNet):
            control_model = controlnet.control_model
            compile_options = gen_compile_options(control_model)
            control_model = control_model.to(load_device)
            controlnet.control_model = oneflow_compile(
                control_model, options=compile_options
            )
            return (controlnet,)
        else:
            print(
                "\033[1;31;40m Warning: {type(controlnet)=} is not ControlLora or ControlNet \033[0m"
            )
            return (controlnet,)


class OneDiffCheckpointLoaderSimple(CheckpointLoaderSimple):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_speedup": (["disable", "enable"],),
            }
        }

    CATEGORY = "OneDiff/Loaders"
    FUNCTION = "onediff_load_checkpoint"

    def onediff_load_checkpoint(
        self, ckpt_name, vae_speedup, output_vae=True, output_clip=True
    ):
        # CheckpointLoaderSimple.load_checkpoint
        modelpatcher, clip, vae = self.load_checkpoint(
            ckpt_name, output_vae, output_clip
        )
        unet_graph_file = generate_graph_path(ckpt_name, modelpatcher.model)
        modelpatcher.model.diffusion_model = compoile_unet(
            modelpatcher.model.diffusion_model, unet_graph_file
        )
        modelpatcher.model._register_state_dict_hook(state_dict_hook)
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

        # set inplace update
        modelpatcher.weight_inplace_update = True
        return modelpatcher, clip, vae


class OneDiffDeepCacheCheckpointLoaderSimple(CheckpointLoaderSimple):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_speedup": (["disable", "enable"],),
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
                    {"default": 1000, "min": 0, "max": 1000, "step": 0.1,},
                ),
            }
        }

    CATEGORY = "OneDiff/Loaders"
    FUNCTION = "onediff_load_checkpoint"

    def onediff_load_checkpoint(
        self,
        ckpt_name,
        vae_speedup,
        output_vae=True,
        output_clip=True,
        static_mode="enable",
        cache_interval=3,
        cache_layer_id=0,
        cache_block_id=1,
        start_step=0,
        end_step=1000,
    ):
        # CheckpointLoaderSimple.load_checkpoint
        modelpatcher, clip, vae = self.load_checkpoint(
            ckpt_name, output_vae, output_clip
        )

        def gen_compile_options(model):
            # cache_key = f'{cache_interval}_{cache_layer_id}_{cache_block_id}_{start_step}_{end_step}'
            graph_file = generate_graph_path(ckpt_name, model)
            return {
                "graph_file": graph_file,
                "graph_file_device": model_management.get_torch_device(),
            }

        if vae_speedup == "enable":
            vae.first_stage_model = oneflow_compile(
                vae.first_stage_model,
                use_graph=True,
                options=gen_compile_options(vae.first_stage_model),
            )

        modelpatcher = deep_cache_speedup(
            model=modelpatcher,
            use_graph=(static_mode == "enable"),
            cache_interval=cache_interval,
            cache_layer_id=cache_layer_id,
            cache_block_id=cache_block_id,
            start_step=start_step,
            end_step=end_step,
            gen_compile_options=gen_compile_options,
        )[0]
        # set inplace update
        modelpatcher.weight_inplace_update = True
        return modelpatcher, clip, vae


class OneDiffQuantCheckpointLoaderSimple(CheckpointLoaderSimple):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_speedup": (["disable", "enable"],),
            }
        }

    CATEGORY = "OneDiff/Loaders"
    FUNCTION = "onediff_load_checkpoint"

    def onediff_load_checkpoint(
        self, ckpt_name, vae_speedup, output_vae=True, output_clip=True
    ):
        # CheckpointLoaderSimple.load_checkpoint
        modelpatcher, clip, vae = self.load_checkpoint(
            ckpt_name, output_vae, output_clip
        )
        ckpt_name = f"{ckpt_name}_quant"
        unet_graph_file = generate_graph_path(ckpt_name, modelpatcher.model)
        load_device = model_management.get_torch_device()
        diffusion_model = modelpatcher.model.diffusion_model.to(load_device)
        unet_model = quantize_model(diffusion_model, inplace=True)
        modelpatcher.model.diffusion_model = unet_model

        offload_device = model_management.unet_offload_device()
        modelpatcher = OneFlowSpeedUpModelPatcher(
            modelpatcher.model,
            load_device=load_device,
            offload_device=offload_device,
            use_graph=True,
            graph_path=unet_graph_file,
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

        # set inplace update
        modelpatcher.weight_inplace_update = True
        return modelpatcher, clip, vae


class OneDiffQuantCheckpointLoaderSimpleAdvanced(CheckpointLoaderSimple):
    @classmethod
    def INPUT_TYPES(s):
        paths = []
        for search_path in folder_paths.get_folder_paths(
            ONEDIFF_QUANTIZED_OPTIMIZED_MODELS
        ):
            if os.path.exists(search_path):
                search_path = Path(search_path)
                paths.extend(
                    [
                        os.path.relpath(p, start=search_path)
                        for p in search_path.glob("*.pt")
                    ]
                )

        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "model_path": (paths,),
                "compile": (["enable", "disable"],),
                "vae_speedup": (["disable", "enable"],),
            }
        }

    CATEGORY = "OneDiff/Loaders"
    FUNCTION = "onediff_load_checkpoint"

    def onediff_load_checkpoint(
        self,
        ckpt_name,
        model_path,
        compile,
        vae_speedup,
        output_vae=True,
        output_clip=True,
    ):
        need_compile = compile == "enable"

        modelpatcher, clip, vae = self.load_checkpoint(
            ckpt_name, output_vae, output_clip
        )
        modelpatcher, vae = onediff_load_quant_checkpoint_advanced(
            ckpt_name, model_path, need_compile, vae_speedup, modelpatcher, vae
        )

        return modelpatcher, clip, vae


class ImageOnlyOneDiffQuantCheckpointLoaderAdvanced(
    comfy_extras.nodes_video_model.ImageOnlyCheckpointLoader
):
    @classmethod
    def INPUT_TYPES(s):
        paths = []
        for search_path in folder_paths.get_folder_paths(
            ONEDIFF_QUANTIZED_OPTIMIZED_MODELS
        ):
            if os.path.exists(search_path):
                search_path = Path(search_path)
                paths.extend(
                    [
                        os.path.relpath(p, start=search_path)
                        for p in search_path.glob("*.pt")
                    ]
                )

        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "model_path": (paths,),
                "compile": (["enable", "disable"],),
                "vae_speedup": (["disable", "enable"],),
            }
        }

    CATEGORY = "OneDiff/Loaders"
    FUNCTION = "onediff_load_checkpoint"

    def onediff_load_checkpoint(
        self,
        ckpt_name,
        model_path,
        compile,
        vae_speedup,
        output_vae=True,
        output_clip=True,
    ):
        need_compile = compile == "enable"
        if need_compile:
            set_boolean_env_var(
                "ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_SCORE_ACCUMULATION_MAX_M", False
            )

        modelpatcher, clip, vae = self.load_checkpoint(
            ckpt_name, output_vae, output_clip
        )
        modelpatcher, vae = onediff_load_quant_checkpoint_advanced(
            ckpt_name, model_path, need_compile, vae_speedup, modelpatcher, vae
        )

        return modelpatcher, clip, vae


if _USE_UNET_INT8:
    from .utils.quant_ksampler_tools import (
        KSampleQuantumBase,
        FineTuneCalibrateInfoMixin,
        SaveQuantizedCalibrateInfoMixin,
        quantized_model_patcher,
        compile_model_patcher_context as _compile_context,
    )

    class QuantKSampler(
        KSampleQuantumBase, FineTuneCalibrateInfoMixin, SaveQuantizedCalibrateInfoMixin
    ):
        @classmethod
        def INPUT_TYPES(s):
            ret = KSampleQuantumBase.INPUT_TYPES()
            ret["required"].update(FineTuneCalibrateInfoMixin.INPUT_TYPES()["required"])
            ret["required"].update(
                SaveQuantizedCalibrateInfoMixin.INPUT_TYPES()["required"]
            )
            ret["required"].update({"static_mode": (["enable", "disable"],)})

            return ret

        RETURN_TYPES = ("LATENT",)
        CATEGORY = "OneDiff/Quant_Tools"
        FUNCTION = "onediff_quant_sample"

        def _get_calibrate_info(
            self, model_patcher, only_compute_density, bits, model_cls, *args, **kwargs
        ):
            model_name = model_patcher.model.__class__.__qualname__
            cache_key = f"{model_name}_{only_compute_density}_{bits}_{model_cls}"
            calibrate_cache = getattr(self, "_calibrate_cache", None)
            if calibrate_cache is None:
                self._calibrate_cache = {}

            calibrate_info = self._calibrate_cache.get(cache_key, None)
            if calibrate_info is None:
                calibrate_info = self.generate_calibrate_info(
                    model_patcher=model_patcher,
                    only_compute_density=only_compute_density,
                    bits=bits,
                    model_cls=model_cls,
                    *args,
                    **kwargs,
                )
                self._calibrate_cache[cache_key] = calibrate_info

            return self._calibrate_cache[cache_key]

        def onediff_quant_sample(
            self,
            onediff_quant,
            quantize_conv,
            quantize_linear,
            conv_mse_threshold,
            linear_mse_threshold,
            compute_density_threshold,
            save_filename_prefix,
            overwrite,
            static_mode,
            bits,
            model,
            *args,
            **kwargs,
        ):
            overwrite = overwrite == "enable"
            only_compute_density = conv_mse_threshold == 1 and linear_mse_threshold == 1
            models_dir = (
                Path(folder_paths.models_dir) / ONEDIFF_QUANTIZED_OPTIMIZED_MODELS
            )
            models_dir.mkdir(parents=True, exist_ok=True)

            model_name = model.model.__class__.__qualname__
            quantize_config_file = (
                models_dir / f"{save_filename_prefix}_{model_name}_quantize_info.pt"
            )
            if quantize_config_file.exists() and not overwrite:
                raise ValueError(
                    f"quantize_config_file {quantize_config_file} exists, please set overwrite=True to overwrite it."
                )

            model_cls = []
            if quantize_conv == "enable":
                model_cls.append(nn.Conv2d)
            if quantize_linear == "enable":
                model_cls.append(nn.Linear)

            calibrate_info = self._get_calibrate_info(
                model_patcher=model,
                only_compute_density=only_compute_density,
                bits=bits,
                model_cls=model_cls,
                *args,
                **kwargs,
            )

            new_calibrate_info = self.fine_tune_calibrate_info(
                model,
                calibrate_info,
                conv_mse_threshold,
                linear_mse_threshold,
                compute_density_threshold,
            )

            self.save_quantized_calibrate_info(
                quantize_config_file, overwrite, new_calibrate_info
            )

            with quantized_model_patcher(
                model_patcher=model,
                layers=list(new_calibrate_info.keys()),
                bits=bits,
                verbose=True,
            ) as qmpatcher:
                if static_mode == "enable":
                    with _compile_context(qmpatcher) as compiled_model:
                        latent_sample = self.generate_latent_sample(
                            compiled_model, *args, **kwargs
                        )
                else:
                    latent_sample = self.generate_latent_sample(
                        qmpatcher, *args, **kwargs
                    )

            return (latent_sample,)
