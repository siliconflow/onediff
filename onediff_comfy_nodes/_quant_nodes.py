from abc import ABC, abstractmethod
import os
import time
import torch
from typing import Any
import numpy as np
import torch.nn as nn
from PIL import Image
from contextlib import contextmanager
from onediff.infer_compiler.utils.module_operations import (
    modify_sub_module,
    get_sub_module,
)
from nodes import KSampler, VAEDecode
from diffusers_quant.utils import find_quantizable_modules
from diffusers_quant.utils import (
    compare_ssim,
    get_quantize_module,
    fake_symm_quantize,
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
    CATEGORY = "OneDiff/Quant"
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
        bits=8,
        quantized_model_generator: callable = lambda x: [x.model.diffusion_model],
        resume: bool = True,
        config_file_path: str = None,
        model_cls=[nn.Linear, nn.Conv2d],
        *args,
        **kwargs,
    ):
        quantize_info_and_relevance_metrics = {}
        if resume and config_file_path:
            if os.path.exists(config_file_path):
                print(f"Resuming from {config_file_path}")
                quantize_info_and_relevance_metrics = torch.load(config_file_path)

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

            length = len(quantizable_modules)
            for index, (sub_name, sub_module) in enumerate(quantizable_modules.items()):
                print(f"Quantizing {index+1}/{length} {sub_name}...")
                if sub_name in quantize_info_and_relevance_metrics:
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

                    quantize_info_and_relevance_metrics[sub_name] = {
                        "scale": scale.to("cpu"),
                        "maxq": maxq,
                        "ssim": ssim,
                        "compute_density": compute_density,
                    }

                    # save
                    if index % 10 == 0 or index == length - 1:
                        torch.save(
                            quantize_info_and_relevance_metrics, config_file_path
                        )

        return images

    @abstractmethod
    def generate_quantized_config(self, vae, model, *args, **kwargs):
        pass


class QuantKSampler(KSampleQuantumBase):
    @classmethod
    def INPUT_TYPES(s):
        ret = KSampleQuantumBase.INPUT_TYPES()
        return ret

    def generate_quantized_config(self, vae, model, *args, **kwargs):
        quantize_config_file = "quantize_info_and_relevance_metrics.pt"
        images = self.quantize_diffusion_model(
            vae=vae,
            model_patcher=model,
            bits=8,
            resume=True,
            config_file_path=quantize_config_file,
            model_cls=[nn.Linear, nn.Conv2d],
            *args,
            **kwargs,
        )
        return images


class DeepCacheQuantKSampler(KSampleQuantumBase):
    @classmethod
    def INPUT_TYPES(s):
        ret = KSampleQuantumBase.INPUT_TYPES()
        return ret

    def generate_quantized_config(self, vae, model, *args, **kwargs):
        model_patcher = model
        bits = 8
        quantize_config_file = "quantize_info_and_relevance_metrics_deep_cache.pt"

        def quantized_model_generator(model_patcher):
            return [model_patcher.deep_cache_unet, model_patcher.fast_deep_cache_unet]

        images = self.quantize_diffusion_model(
            vae=vae,
            model_patcher=model_patcher,
            bits=bits,
            resume=True,
            config_file_path=quantize_config_file,
            model_cls=[nn.Linear, nn.Conv2d],
            quantized_model_generator=quantized_model_generator,
            *args,
            **kwargs,
        )
        return images
