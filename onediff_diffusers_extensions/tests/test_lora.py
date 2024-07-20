import pytest
import random
from PIL import Image
from collections import defaultdict
from typing import Dict, List, Tuple
from pathlib import Path

import torch
from torch import Tensor
import numpy as np
import safetensors.torch
from skimage.metrics import structural_similarity
from diffusers import DiffusionPipeline
from onediff.infer_compiler import oneflow_compile

from onediffx.lora import load_and_fuse_lora, unfuse_lora, set_and_fuse_adapters, get_active_adapters, delete_adapters

HEIGHT = 1024
WIDTH = 1024
NUM_STEPS = 30
LORA_SCALE = 0.5
LATENTS = torch.randn(1, 4, 128, 128, generator=torch.cuda.manual_seed(0), dtype=torch.float16, device="cuda")

image_file_prefix = "/share_nfs/onediff_ci/diffusers/images/1.0"

@pytest.fixture
def prepare_loras() -> Dict[str, Dict[str, Tensor]]:
    loras = [
        "/share_nfs/onediff_ci/diffusers/loras/SDXL-Emoji-Lora-r4.safetensors",
        "/share_nfs/onediff_ci/diffusers/loras/sdxl_metal_lora.safetensors",
        "/share_nfs/onediff_ci/diffusers/loras/simple_drawing_xl_b1-000012.safetensors",
        "/share_nfs/onediff_ci/diffusers/loras/texta.safetensors",
        "/share_nfs/onediff_ci/diffusers/loras/watercolor_v1_sdxl_lora.safetensors",
    ]
    loras = {x: safetensors.torch.load_file(x) for x in loras}
    return loras

@pytest.fixture
def get_loras(prepare_loras) -> Dict[str, Dict[str, Tensor]]:
    def _get_loras():
        return {name: lora_dict.copy() for name, lora_dict in prepare_loras.items()}
    return _get_loras

@pytest.fixture
def get_multi_loras(prepare_loras) -> Dict[str, Dict[str, Tensor]]:
    def _get_multi_loras():
        multi_lora = {}
        current_name = []
        current_lora = []
        for name, lora_dict in prepare_loras.items():
            current_name.append(name)
            current_lora.append(lora_dict)
            multi_lora[tuple(current_name)] = current_lora
        return multi_lora
    return _get_multi_loras


@pytest.fixture
def pipe():
    MODEL_ID = "/share_nfs/hf_models/stable-diffusion-xl-base-1.0"
    pipeline = DiffusionPipeline.from_pretrained(
        MODEL_ID, variant="fp16", torch_dtype=torch.float16
    ).to("cuda")
    return pipeline

def generate_image(pipe):
    image = pipe(
        "masterpiece, best quality, mountain",
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        generator=torch.manual_seed(0),
        latents=LATENTS.clone(),
    ).images[0]
    return image

def prepare_target_images(pipe, loras):
    target_images_list = [
        f"{image_file_prefix}/test_sdxl_lora_{str(Path(name).stem)}_{HEIGHT}_{WIDTH}.png" for name in loras
    ]
    if all(Path(x).exists() for x in target_images_list):
        return

    print("Didn't find target images, try to generate...")
    for name, lora in loras.items():
        pipe.load_lora_weights(lora.copy())
        pipe.fuse_lora()
        image = generate_image(pipe)
        pipe.unfuse_lora()
        pipe.unload_lora_weights()
        image.save(f"{image_file_prefix}/test_sdxl_lora_{str(Path(name).stem)}_{HEIGHT}_{WIDTH}.png")
    torch.cuda.empty_cache()

def prepare_target_images_multi_lora(pipe, loras, multi_loras):
    target_images_list = [
        f"{image_file_prefix}/test_sdxl_multi_lora_{'_'.join([str(Path(name).stem) for name in names])}_{HEIGHT}_{WIDTH}.png" for names in multi_loras
    ]
    if all(Path(x).exists() for x in target_images_list):
        return

    for name, lora in loras.items():
        print(f"loading name: {name}")
        pipe.load_lora_weights(lora.copy(), adapter_name=str(Path(name).stem))

    print("Didn't find target images, try to generate...")
    for names, loras in multi_loras.items():
        names = [str(Path(name).stem) for name in names]
        pipe.set_adapters(names, [LORA_SCALE, ] * len(names))
        image = generate_image(pipe)
        image_name = f"{image_file_prefix}/test_sdxl_multi_lora_{'_'.join([str(Path(name).stem) for name in names])}_{HEIGHT}_{WIDTH}.png" 
        image.save(image_name)
    pipe.unload_lora_weights()
    torch.cuda.empty_cache()


def preload_multi_loras(pipe, loras):
    for name, lora in loras.items():
        load_and_fuse_lora(
            pipe, lora.copy(), adapter_name=Path(name).stem,
        )
        unfuse_lora(pipe)


def test_lora_loading(pipe, get_loras):
    pipe.unet = oneflow_compile(pipe.unet)
    pipe("a cat", height=HEIGHT, width=WIDTH, num_inference_steps=NUM_STEPS,).images[0]
    loras = get_loras()
    prepare_target_images(pipe, loras)

    for name, lora in loras.items():
        load_and_fuse_lora(pipe, lora.copy())
        images_fusion = generate_image(pipe)
        target_image = np.array(
            Image.open(f"{image_file_prefix}/test_sdxl_lora_{str(Path(name).stem)}_{HEIGHT}_{WIDTH}.png")
        )
        curr_image = np.array(images_fusion)
        ssim = structural_similarity(
            curr_image, target_image, channel_axis=-1, data_range=255
        )
        unfuse_lora(pipe)
        images_fusion.save(f"./test_sdxl_lora_{str(Path(name).stem)}_{HEIGHT}_{WIDTH}.png")
        print(f"lora {name} ssim {ssim}")
        assert ssim > 0.92, f"LoRA {name} ssim too low"


def test_multi_lora_loading(pipe, get_multi_loras, get_loras):
    pipe.unet = oneflow_compile(pipe.unet)
    multi_loras = get_multi_loras()
    loras = get_loras()
    prepare_target_images_multi_lora(pipe, loras, multi_loras)
    preload_multi_loras(pipe, loras)

    for names, loras in multi_loras.items():
        names = [str(Path(name).stem) for name in names]
        set_and_fuse_adapters(pipe, names, [LORA_SCALE, ] * len(names))

        images_fusion = generate_image(pipe)
        image_name = '_'.join([str(Path(name).stem) for name in names])
        target_image = np.array(
            Image.open(f"{image_file_prefix}/test_sdxl_multi_lora_{image_name}_{HEIGHT}_{WIDTH}.png")
        )
        images_fusion.save(f"./test_sdxl_multi_lora_{image_name}_{HEIGHT}_{WIDTH}.png")
        curr_image = np.array(images_fusion)
        ssim = structural_similarity(
            curr_image, target_image, channel_axis=-1, data_range=255
        )
        print(f"lora {names} ssim {ssim}")
        assert ssim > 0.92, f"LoRA {names} ssim too low"


def test_get_active_adapters(pipe, get_multi_loras):
    multi_loras = get_multi_loras()
    for names, _ in multi_loras.items():
        names = [str(Path(name).stem) for name in names]
        set_and_fuse_adapters(pipe, names)
        active_adapters = get_active_adapters(pipe)
        print(f"current adapters: {active_adapters}, target adapters: {names}")
        assert set(active_adapters) == set(names)


def test_delete_adapters(pipe, get_multi_loras):
    multi_loras = get_multi_loras()
    for names, _ in multi_loras.items():
        names = [str(Path(name).stem) for name in names]
        names_to_delete = random.sample(names, k=random.randint(0, len(names)))
        set_and_fuse_adapters(pipe, names)
        delete_adapters(pipe, names_to_delete)
        active_adapters = get_active_adapters(pipe)
        print(f"current adapters: {active_adapters}, target adapters: {list(set(names) - set(names_to_delete))}")
        assert set(active_adapters) == set(names) - set(names_to_delete)
