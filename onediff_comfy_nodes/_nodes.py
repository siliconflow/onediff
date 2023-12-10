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

from .utils import OneFlowSpeedUpModelPatcher, save_graph, load_graph, OUTPUT_FOLDER

__all__ = [
    "ModelSpeedup",
    "ModelGraphLoader",
    "ModelGraphSaver",
    "UNETLoaderInt8",
    "VaeSpeedup",
    "VaeGraphLoader",
    "VaeGraphSaver",
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
        from .utils import replace_module_with_quantizable_module

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
        from .utils import quantized_and_save_model

        diffusion_model = model.model.diffusion_model
        output_dir = os.path.join(folder_paths.models_dir, "unet_int8", output_dir)
        is_quantize_conv = conv == "enable"
        is_quantize_linear = linear == "enable"
        quantized_and_save_model(
            diffusion_model,
            output_dir,
            quantize_conv=is_quantize_conv,
            quantize_linear=is_quantize_linear,
            verbose=False,
        )
        return {}
