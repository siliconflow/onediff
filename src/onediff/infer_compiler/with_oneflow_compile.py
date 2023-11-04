from ._config import *
from .utils import replace_module_with_quantizable_module

from copy import copy

import comfy
import folder_paths
from comfy import model_management
import comfy.model_patcher
from pathlib import Path


class OneFlowSpeedUpModelPatcher(comfy.model_patcher.ModelPatcher):
    def __init__(self, model, load_device, offload_device, size=0, current_device=None, static_mode=None):
        from onediff.infer_compiler import oneflow_compile
        use_graph = static_mode == "enable"
        self.size = size
        self.model = copy(model)
        self.model.__dict__['_modules'] = copy(model.__dict__['_modules'])
        self.model.__dict__['_modules']['diffusion_model'] = oneflow_compile(
            self.model.diffusion_model, use_graph=use_graph
        )
        self.patches = {}
        self.backup = {}
        self.model_options = {"transformer_options":{}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device
        if current_device is None:
            self.current_device = self.offload_device
        else:
            self.current_device = current_device


class ComfySpeedup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "static_mode": (["enable", "disable"],),
            },
        }

    RETURN_TYPES = ("MODEL","VAE","CLIP")
    FUNCTION = "speedup"
    CATEGORY = "utils"

    def speedup(self, model, vae, clip, static_mode):
        from onediff.infer_compiler import oneflow_compile
        offload_device = model_management.unet_offload_device()
        oneflow_model = OneFlowSpeedUpModelPatcher(
            model.model,
            load_device=model_management.get_torch_device(),
            offload_device=offload_device,
            static_mode=static_mode,
        )

        vae.vae_dtype = torch.float32 
        vae.first_stage_model.to(torch.float32)
        vae.first_stage_model = oneflow_compile(vae.first_stage_model, use_graph=True)
        # TODO compile clip...

        return (oneflow_model,vae,clip,)


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

    CATEGORY = "advanced/loaders"

    def load_unet_int8(self, model_path):
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
