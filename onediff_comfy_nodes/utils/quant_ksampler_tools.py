import time
from typing import Dict
import os
import torch
import torch.nn as nn

import torch

# onediff
from onediff.infer_compiler.utils.module_operations import (
    modify_sub_module,
    get_sub_module,
)


# ComfyUI
import folder_paths
from nodes import KSampler, VAEDecode

# onediff_quant
from onediff_quant.utils import find_quantizable_modules
from onediff_quant.utils import (
    get_quantize_module,
    symm_quantize,
    metric_quantize_costs,
)
from onediff_quant import Quantizer


def quantize_sub_module(model, sub_name, sub_module, bits):
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

    def restore():
        sub_module.weight.data = org_weight_data
        sub_module.weight.requires_grad = org_requires_grad
        modify_sub_module(model, sub_name, sub_module)

    return restore


class quantized_model_patcher:
    def __init__(self, model_patcher, layers, bits, verbose=False):
        self.model_patcher = model_patcher
        self.layers = layers
        self.bits = bits
        self.diffusion_model = model_patcher.model.diffusion_model
        self.handles = []

        self.verbose = verbose
        self.conv_count = 0
        self.linear_count = 0

    def __enter__(self):
        self.start_time = time.time()
        for sub_name in self.layers:
            sub_module = get_sub_module(self.diffusion_model, sub_name)
            if isinstance(sub_module, nn.Conv2d):
                self.conv_count += 1
            elif isinstance(sub_module, nn.Linear):
                self.linear_count += 1
            self.handles.append(
                quantize_sub_module(
                    self.diffusion_model, sub_name, sub_module, bits=self.bits
                )
            )
        return self.model_patcher

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle()

        if self.verbose:
            end = time.time()
            print(f"\n=> Total time: {end - self.start_time:.4f} seconds")
            print(f"Quantized {self.conv_count} convs, {self.linear_count} linears.\n")
        return False


class KSampleQuantumBase(KSampler):
    @classmethod
    def INPUT_TYPES(s):
        ret = KSampler.INPUT_TYPES()

        ret["required"].update(
            {"onediff_quant": ("STRING", {"default": "Quantization Settings"})}
        )

        ret["required"].update(
            {"process_cache_file_prefix": ("STRING", {"default": "unet"})}
        )
        ret["required"].update({"bits": ("INT", {"default": 8, "min": 8, "max": 8})})
        ret["required"].update({"quantize_conv": (["enable", "disable"],)})
        ret["required"].update({"quantize_linear": (["enable", "disable"],)})
        return ret

    CATEGORY = "OneDiff/Quant_Tools"

    def generate_latent_sample(self, model_patcher, *args, **kwargs):
        # self.sample from KSampler
        latent_samples = self.sample(model_patcher, *args, **kwargs)
        assert (
            len(latent_samples) == 1
        ), "Only batch size 1 is supported for quantization"
        out = latent_samples[0]
        return out

    def generate_pipeline(self, model_patcher, *args, **kwargs):
        generate_img = self.generate_latent_sample

        class ImageGenerationPipeline:
            def __init__(self):
                self.model = model_patcher
                self.args = args
                self.kwargs = kwargs

            def __call__(self, *args, **kwargs):
                return generate_img(self.model, *self.args, **self.kwargs)

        return ImageGenerationPipeline()

    def generate_calibrate_info(
        self,
        model_patcher,
        only_compute_density: bool = False,
        bits=8,
        quantized_model_generator: callable = lambda x: x.model.diffusion_model,
        resume: bool = True,
        cached_process_output_path: str = None,
        model_cls=[nn.Linear, nn.Conv2d],
        *args,
        **kwargs,
    ) -> Dict:
        """return calibrate_info"""

        calibrate_info = {}
        if resume and cached_process_output_path:
            if os.path.exists(cached_process_output_path):
                print(f"Resuming from {cached_process_output_path}")
                calibrate_info = torch.load(cached_process_output_path)

        diffusion_model = quantized_model_generator(model_patcher)

        quantizable_modules = find_quantizable_modules(
            diffusion_model, module_cls=model_cls
        )

        if set(quantizable_modules.keys()) == set(calibrate_info.keys()):
            return calibrate_info

        org_latent_sample = self.generate_latent_sample(model_patcher, *args, **kwargs)[
            "samples"
        ]
        pipe = self.generate_pipeline(model_patcher, *args, **kwargs)
        quantize_costs = metric_quantize_costs(
            pipe, pipe_kwargs={}, quantizable_modules=quantizable_modules
        )

        if only_compute_density:
            for sub_name, _ in quantizable_modules.items():
                compute_density = quantize_costs.get_compute_density(sub_name)
                calibrate_info[sub_name] = {
                    "compute_density": compute_density,
                }

            torch.save(calibrate_info, cached_process_output_path)
            return calibrate_info

        length = len(quantizable_modules)
        for index, (sub_name, _) in enumerate(quantizable_modules.items()):
            print(f"Quantizing {index+1}/{length} {sub_name}...")
            if sub_name in calibrate_info:
                continue

            with quantized_model_patcher(
                model_patcher=model_patcher, layers=[sub_name], bits=bits
            ) as qmpatcher:
                cur_latent_sample = self.generate_latent_sample(
                    qmpatcher, *args, **kwargs
                )["samples"]

                mse = torch.mean((org_latent_sample - cur_latent_sample) ** 2)

                similarity_mse = 1 - mse
                compute_density = quantize_costs.get_compute_density(sub_name)
                print(
                    f"similarity_mse: {similarity_mse:.4f}, compute_density: {compute_density:.4f}"
                )

                calibrate_info[sub_name] = {
                    "ssim": similarity_mse,
                    "compute_density": compute_density,
                }

                # save
                if index % 10 == 0 or index == length - 1:
                    torch.save(calibrate_info, cached_process_output_path)

        return calibrate_info


class FineTuneCalibrateInfoMixin:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
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
                        "default": 0,
                        "min": 0,
                        "max": 100000,
                        "step": 1,
                        "display": "number",
                    },
                ),
            },
        }

    def fine_tune_calibrate_info(
        self,
        model,
        calibrate_info,
        conv_ssim_threshold,
        linear_ssim_threshold,
        compute_density_threshold,
    ):
        new_calibrate_info = {}
        for sub_name, sub_info in calibrate_info.items():
            if sub_info["compute_density"] < compute_density_threshold:
                continue

            sub_model = get_sub_module(model.model.diffusion_model, sub_name)

            if isinstance(sub_model, nn.Conv2d) and "ssim" in sub_info:
                if sub_info["ssim"] < conv_ssim_threshold:
                    continue

            if isinstance(sub_model, nn.Linear) and "ssim" in sub_info:
                if sub_info["ssim"] < linear_ssim_threshold:
                    continue

            new_calibrate_info[sub_name] = sub_info

        return new_calibrate_info


class SaveQuantizedCalibrateInfoMixin:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_filename_prefix": ("STRING", {"default": "unet"}),
                "overwrite": (["enable", "disable"],),
            },
        }

    def save_quantized_calibrate_info(
        self, quantize_config_file, overwrite, calibrate_info
    ):
        if not overwrite:
            if quantize_config_file.exists():
                print(f"Quantize config file {quantize_config_file} exists, skip.")
                return

        torch.save(calibrate_info, quantize_config_file)
        print(f"Saved quantize config file to {quantize_config_file}")

        return quantize_config_file
