from typing import Dict
from ._config import ONEDIFF_QUANTIZED_OPTIMIZED_MODELS
from abc import ABC, abstractmethod
import os
import copy
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from pathlib import Path
from contextlib import contextmanager

# onediff
from onediff.infer_compiler.utils.module_operations import (
    modify_sub_module,
    get_sub_module,
)
from onediff.optimization.quant_optimizer import quantize_model
from onediff.infer_compiler import oneflow_compile

# ComfyUI
import folder_paths
from nodes import KSampler, VAEDecode

# diffusers_quant
from diffusers_quant.utils import find_quantizable_modules
from diffusers_quant.utils import (
    compare_ssim,
    get_quantize_module,
    symm_quantize,
    metric_quantize_costs,
)
from diffusers_quant import Quantizer
from diffusers_quant.utils import *


def format_image(image) -> Image:
    """Converts a torch tensor to a PIL image."""
    if image.shape[0] == 1:
        image = image.squeeze(0)

    i = 255.0 * image.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img


@contextmanager
def quantize_sub_module_context(model, sub_name, sub_module=None, bits=8):
    if sub_module is None:
        sub_module = get_sub_module(model, sub_name)

    quantizer = Quantizer()
    quantizer.configure(bits=bits, perchannel=True)
    quantizer.find_params(sub_module.weight.float(), weight=True)
    shape = [-1] + [1] * (len(sub_module.weight.shape) - 1)
    scale = quantizer.scale.reshape(*shape)

    org_weight_data = sub_module.weight.data
    org_requires_grad = sub_module.weight.requires_grad

    # save_as_float = False
    sub_module.weight.requires_grad = False
    sub_module.weight.data = symm_quantize(
        sub_module.weight.data, scale.to(sub_module.weight.data.device), quantizer.maxq
    )
    input_scale_and_zero_point = [None, None]
    quant_module = get_quantize_module(
        sub_module,
        sub_name,
        input_scale_and_zero_point + [scale.reshape(-1).tolist()],
        False,  # fake_quant
        False,
        bits,
    )

    modify_sub_module(model, sub_name, quant_module)
    yield scale, quantizer.maxq

    sub_module.weight.data = org_weight_data
    sub_module.weight.requires_grad = org_requires_grad
    modify_sub_module(model, sub_name, sub_module)


class KSampleQuantumBase(ABC, KSampler, VAEDecode):
    @classmethod
    def INPUT_TYPES(s):
        ret = KSampler.INPUT_TYPES()
        ret["required"].update({"vae": ("VAE",)})
        return ret

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "OneDiff/Quant_Tools"
    FUNCTION = "generate_quantized_config"

    def generate_img(self, vae, model, *args, **kwargs):
        # self.sample from KSampler
        samples = self.sample(model, *args, **kwargs)
        assert (
            len(samples[0]["samples"]) == 1
        ), "Only batch size 1 is supported for quantization"

        # self.decode from VAEDecode
        images = self.decode(vae=vae, samples=samples[0])

        return images

    def generate_pipeline(self, vae, model, *args, **kwargs):
        generate_img = self.generate_img

        class ImageGenerationPipeline:
            def __init__(self):
                self.vae = vae
                self.model = model
                self.args = args
                self.kwargs = kwargs

            def __call__(self, *args, **kwargs):
                return generate_img(self.vae, self.model, *self.args, **self.kwargs)

        return ImageGenerationPipeline()

    def quantize_diffusion_model(
        self,
        vae,
        model_patcher,
        only_compute_density: bool = False,
        bits=8,
        quantized_model_generator: callable = lambda x: [x.model.diffusion_model],
        resume: bool = True,
        config_file_path: str = None,
        model_cls=[nn.Linear, nn.Conv2d],
        *args,
        **kwargs,
    ) -> Dict:
        """return calibrate_info"""
        calibrate_info = {}
        if resume and config_file_path:
            if os.path.exists(config_file_path):
                print(f"Resuming from {config_file_path}")
                calibrate_info = torch.load(config_file_path)

        images = self.generate_img(vae, model_patcher, *args, **kwargs)
        original_image = format_image(images[0])

        for diffusion_model in quantized_model_generator(model_patcher):
            quantizable_modules = find_quantizable_modules(
                diffusion_model, module_cls=model_cls
            )

            pipe = self.generate_pipeline(vae, model_patcher, *args, **kwargs)
            quantize_costs = metric_quantize_costs(
                pipe, pipe_kwargs={}, quantizable_modules=quantizable_modules
            )

            if only_compute_density:
                for sub_name, sub_module in quantizable_modules.items():
                    compute_density = quantize_costs.get_compute_density(sub_name)
                    calibrate_info[sub_name] = {
                        "compute_density": compute_density,
                    }
                torch.save(calibrate_info, config_file_path)

            length = len(quantizable_modules)
            for index, (sub_name, sub_module) in enumerate(quantizable_modules.items()):
                print(f"Quantizing {index+1}/{length} {sub_name}...")
                if sub_name in calibrate_info:
                    continue

                with quantize_sub_module_context(
                    model=diffusion_model,
                    sub_name=sub_name,
                    sub_module=sub_module,
                    bits=bits,
                ) as (scale, maxq):
                    quantized_images = self.generate_img(
                        vae, model_patcher, *args, **kwargs
                    )
                    quantized_image = format_image(quantized_images[0])
                    ssim = compare_ssim(
                        np.asarray(original_image),
                        np.asarray(quantized_image),
                        channel_axis=2,
                    )
                    compute_density = quantize_costs.get_compute_density(sub_name)
                    print(f"SSIM for {sub_name}: {ssim} density: {compute_density}")

                    calibrate_info[sub_name] = {
                        "scale": scale.to("cpu"),
                        "maxq": maxq,
                        "ssim": ssim,
                        "compute_density": compute_density,
                    }

                    # save
                    if index % 10 == 0 or index == length - 1:
                        torch.save(calibrate_info, config_file_path)

        return images[0], calibrate_info

    @abstractmethod
    def generate_quantized_config(self, vae, model, *args, **kwargs):
        pass


class UnetQuantKSampler(KSampleQuantumBase):
    @classmethod
    def INPUT_TYPES(s):
        ret = KSampleQuantumBase.INPUT_TYPES()
        ret["required"].update(
            {"fastquant_model_prefix": ("STRING", {"default": "unet"})}
        )
        ret["required"].update({"only_compute_density": (["disable", "enable"],)})
        ret["required"].update({"quantize_conv": (["enable", "disable"],)})
        ret["required"].update({"quantize_linear": (["enable", "disable"],)})
        return ret

    RETURN_TYPES = (
        "IMAGE",
        "CALIBRATE_INFO",
    )

    def generate_quantized_config(
        self,
        vae,
        fastquant_model_prefix,
        only_compute_density,
        quantize_conv,
        quantize_linear,
        model,
        *args,
        **kwargs,
    ):
        models_dir = Path(folder_paths.models_dir) / ONEDIFF_QUANTIZED_OPTIMIZED_MODELS
        models_dir.mkdir(parents=True, exist_ok=True)
        model_name = model.model.__class__.__qualname__
        quantize_config_file = (
            models_dir / f"{fastquant_model_prefix}_{model_name}_quantize_info.pt"
        )

        model_cls = []
        if quantize_conv == "enable":
            model_cls.append(nn.Conv2d)
        if quantize_linear == "enable":
            model_cls.append(nn.Linear)

        image, calibrate_info = self.quantize_diffusion_model(
            vae=vae,
            model_patcher=model,
            only_compute_density=(only_compute_density == "enable"),
            bits=8,
            resume=True,
            config_file_path=quantize_config_file,
            model_cls=model_cls,
            *args,
            **kwargs,
        )
        return (
            image,
            calibrate_info,
        )


class FineTuneCalibrateInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "calibrate_info": ("CALIBRATE_INFO",),
                "conv_ssim_threshold": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0,
                        "max": 1,
                        "step": 0.01,
                        "display": "number",
                    },
                ),
                "linear_ssim_threshold": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0,
                        "max": 1,
                        "step": 0.01,
                        "display": "number",
                    },
                ),
                "compute_density_threshold": (
                    "INT",
                    {
                        "default": 100,
                        "min": 0,
                        "max": 10000,
                        "step": 10,
                        "display": "number",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "MODEL",
        "CALIBRATE_INFO",
    )
    FUNCTION = "fine_tune_calibrate_info"
    CATEGORY = "OneDiff/Quant_Tools"

    def _new_modelpatcher(self, model):
        modelpatcher = getattr(model, "modelpatcher", None)
        if modelpatcher is None:
            modelpatcher = model.clone()

        modelpatcher.model = copy.copy(model.model)
        modelpatcher.model.__dict__["_modules"] = copy.copy(
            model.model.__dict__["_modules"]
        )
        self.modelpatcher = modelpatcher
        return modelpatcher

    def fine_tune_calibrate_info(
        self,
        model,
        calibrate_info,
        conv_ssim_threshold,
        linear_ssim_threshold,
        compute_density_threshold,
    ):
        modelpatcher = self._new_modelpatcher(model)

        new_calibrate_info = {}
        for sub_name, sub_info in calibrate_info.items():
            if sub_info["compute_density"] < compute_density_threshold:
                continue

            sub_model = get_sub_module(modelpatcher.model.diffusion_model, sub_name)

            if isinstance(sub_model, nn.Conv2d):
                if sub_info["ssim"] < conv_ssim_threshold:
                    continue

            if isinstance(sub_model, nn.Linear):
                if sub_info["ssim"] < linear_ssim_threshold:
                    continue

            new_calibrate_info[sub_name] = sub_info

        diffusion_model = modelpatcher.model.diffusion_model

        quant_module = quantize_model(
            diffusion_model, inplace=False, calibrate_info=new_calibrate_info
        )

        modelpatcher.model.diffusion_model = quant_module
        return (modelpatcher, new_calibrate_info)


class LoadQuantizedConfig:
    @classmethod
    def INPUT_TYPES(s):
        paths = folder_paths.get_filename_list(ONEDIFF_QUANTIZED_OPTIMIZED_MODELS)
        return {
            "required": {
                "model_name": (paths,),
            },
        }

    RETURN_TYPES = ("CALIBRATE_INFO",)
    FUNCTION = "load_quantized_config"
    CATEGORY = "OneDiff/Quant_Tools"

    def load_quantized_config(self, model_name):
        models_dir = Path(folder_paths.models_dir) / ONEDIFF_QUANTIZED_OPTIMIZED_MODELS
        quantize_config_file = models_dir / model_name
        calibrate_info = torch.load(quantize_config_file)
        return (calibrate_info,)


class SaveQuantizedConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "calibrate_info": ("CALIBRATE_INFO",),
                "model_name": ("STRING", {"default": "unet"}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_quantized_config"
    CATEGORY = "OneDiff/Quant_Tools"

    def save_quantized_config(self, model_name, calibrate_info):
        model_name = f"{model_name}_calibrate_info.pt"
        models_dir = Path(folder_paths.models_dir) / ONEDIFF_QUANTIZED_OPTIMIZED_MODELS
        quantize_config_file = models_dir / model_name

        torch.save(calibrate_info, quantize_config_file)

        return {"ui": {"model_name": model_name}}
