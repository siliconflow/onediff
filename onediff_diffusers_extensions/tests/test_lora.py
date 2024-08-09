import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
import safetensors.torch

import torch
from diffusers import DiffusionPipeline
from onediff.infer_compiler import oneflow_compile

from onediffx.lora import (
    delete_adapters,
    get_active_adapters,
    load_and_fuse_lora,
    load_lora_and_optionally_fuse,
    set_and_fuse_adapters,
    unfuse_lora,
)
from PIL import Image
from skimage.metrics import structural_similarity

HEIGHT = 1024
WIDTH = 1024
NUM_STEPS = 30
LORA_SCALE = 0.5
LATENTS = torch.randn(
    1,
    4,
    128,
    128,
    generator=torch.cuda.manual_seed(0),
    dtype=torch.float16,
    device="cuda",
)

image_file_prefix = "/share_nfs/onediff_ci/diffusers/images/1.0"


@pytest.fixture
def prepare_loras() -> Dict[str, Dict[str, torch.Tensor]]:
    loras = [
        "/share_nfs/onediff_ci/diffusers/loras/SDXL-Emoji-Lora-r4.safetensors",
        "/share_nfs/onediff_ci/diffusers/loras/sdxl_metal_lora.safetensors",
        "/share_nfs/onediff_ci/diffusers/loras/simple_drawing_xl_b1-000012.safetensors",
        "/share_nfs/onediff_ci/diffusers/loras/texta.safetensors",
        "/share_nfs/onediff_ci/diffusers/loras/watercolor_v1_sdxl_lora.safetensors",
    ]
    loras = {Path(x).stem: safetensors.torch.load_file(x) for x in loras}
    return loras


@pytest.fixture
def get_loras(prepare_loras) -> Dict[str, Dict[str, torch.Tensor]]:
    def _get_loras():
        return {name: lora_dict.copy() for name, lora_dict in prepare_loras.items()}

    return _get_loras


@pytest.fixture
def get_multi_loras(prepare_loras) -> Dict[str, Dict[str, torch.Tensor]]:
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


def get_pipe(name: str = "sdxl"):
    id_dict = {
        "sd1.5": "/share_nfs/hf_models/stable-diffusion-v1-5",
        "sdxl": "/share_nfs/hf_models/stable-diffusion-xl-base-1.0",
        "sd2.1": "/share_nfs/hf_models/stable-diffusion-2-1",
    }
    MODEL_ID = id_dict[name]
    pipeline = DiffusionPipeline.from_pretrained(
        MODEL_ID, variant="fp16", torch_dtype=torch.float16
    ).to("cuda")
    return pipeline


@pytest.fixture(scope="session")
def pipe():
    pipeline = get_pipe()
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
        f"{image_file_prefix}/test_sdxl_lora_{name}_{HEIGHT}_{WIDTH}.png"
        for name in loras
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
        image.save(f"{image_file_prefix}/test_sdxl_lora_{name}_{HEIGHT}_{WIDTH}.png")
    torch.cuda.empty_cache()


def prepare_target_images_multi_lora(pipe, loras, multi_loras):
    target_images_list = [
        f"{image_file_prefix}/test_sdxl_multi_lora_{'_'.join(names)}_{HEIGHT}_{WIDTH}.png"
        for names in multi_loras
    ]
    if all(Path(x).exists() for x in target_images_list):
        return

    for name, lora in loras.items():
        print(f"loading name: {name}")
        pipe.load_lora_weights(lora.copy(), adapter_name=name)

    print("Didn't find target images, try to generate...")
    for names, loras in multi_loras.items():
        pipe.set_adapters(
            names,
            [
                LORA_SCALE,
            ]
            * len(names),
        )
        image = generate_image(pipe)
        image_name = f"{image_file_prefix}/test_sdxl_multi_lora_{'_'.join(names)}_{HEIGHT}_{WIDTH}.png"
        image.save(image_name)
    pipe.unload_lora_weights()
    torch.cuda.empty_cache()


def preload_multi_loras(pipe, loras):
    for name, lora in loras.items():
        load_lora_and_optionally_fuse(
            pipe,
            lora.copy(),
            adapter_name=name,
            fuse=False,
        )
        unfuse_lora(pipe)


def test_lora_loading(pipe, get_loras):
    pipe.unet = oneflow_compile(pipe.unet)
    pipe(
        "a cat",
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
    ).images[0]
    loras = get_loras()
    prepare_target_images(pipe, loras)

    for name, lora in loras.items():
        load_and_fuse_lora(pipe, lora.copy())
        images_fusion = generate_image(pipe)
        target_image = np.array(
            Image.open(
                f"{image_file_prefix}/test_sdxl_lora_{name}_{HEIGHT}_{WIDTH}.png"
            )
        )
        curr_image = np.array(images_fusion)
        ssim = structural_similarity(
            curr_image, target_image, channel_axis=-1, data_range=255
        )
        unfuse_lora(pipe)
        images_fusion.save(f"./test_sdxl_lora_{name}_{HEIGHT}_{WIDTH}.png")

        print(f"lora {name} ssim {ssim}")
        assert ssim > 0.92, f"LoRA {name} ssim too low"


def test_multi_lora_loading(pipe, get_multi_loras, get_loras):
    pipe.unet = oneflow_compile(pipe.unet)
    multi_loras = get_multi_loras()
    loras = get_loras()
    prepare_target_images_multi_lora(pipe, loras, multi_loras)
    preload_multi_loras(pipe, loras)

    for names, loras in multi_loras.items():
        set_and_fuse_adapters(
            pipe,
            names,
            [
                LORA_SCALE,
            ]
            * len(names),
        )

        images_fusion = generate_image(pipe)
        image_name = "_".join(names)
        target_image = np.array(
            Image.open(
                f"{image_file_prefix}/test_sdxl_multi_lora_{image_name}_{HEIGHT}_{WIDTH}.png"
            )
        )
        images_fusion.save(f"./test_sdxl_multi_lora_{image_name}_{HEIGHT}_{WIDTH}.png")
        curr_image = np.array(images_fusion)
        ssim = structural_similarity(
            curr_image, target_image, channel_axis=-1, data_range=255
        )
        print(f"lora {names} ssim {ssim}")
        assert ssim > 0.92, f"LoRA {names} ssim too low"


def test_get_active_adapters(pipe, get_multi_loras, get_loras):
    multi_loras = get_multi_loras()
    preload_multi_loras(pipe, get_loras())
    for names, _ in multi_loras.items():
        set_and_fuse_adapters(pipe, names)
        active_adapters = get_active_adapters(pipe)
        print(f"current adapters: {active_adapters}, target adapters: {names}")
        assert set(active_adapters) == set(names)


def test_delete_adapters(pipe, get_multi_loras, get_loras):
    multi_loras = get_multi_loras()
    for names, _ in multi_loras.items():
        preload_multi_loras(pipe, get_loras())
        names_to_delete = random.sample(names, k=random.randint(0, len(names)))
        set_and_fuse_adapters(pipe, names)
        delete_adapters(pipe, names_to_delete)
        active_adapters = get_active_adapters(pipe)
        print(
            f"current adapters: {active_adapters}, target adapters: {list(set(names) - set(names_to_delete))}"
        )
        assert set(active_adapters) == set(names) - set(names_to_delete)


def test_lora_numerical_stability():
    original_pipe = get_pipe("sd1.5")
    pipe = get_pipe("sd1.5")
    loras = [
        "/share_nfs/onediff_ci/diffusers/loras/SD15-IllusionDiffusionPattern-LoRA.safetensors",
        "/share_nfs/onediff_ci/diffusers/loras/SD15-Megaphone-LoRA.safetensors",
    ]
    loras = {Path(x).stem: safetensors.torch.load_file(x) for x in loras}
    param_name = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"

    for _ in range(1000):
        for name, lora in loras.items():
            load_lora_and_optionally_fuse(
                pipe, lora.copy(), adapter_name=name, fuse=False
            )
        set_and_fuse_adapters(
            pipe, adapter_names=list(loras.keys()), adapter_weights=[0.2, 0.2]
        )
        for name in loras:
            delete_adapters(pipe, name, safe_delete=True)

    assert torch.allclose(
        original_pipe.unet.get_parameter(param_name),
        pipe.unet.get_parameter(param_name),
        rtol=0,
        atol=1e-3,
    )
    print(
        f"numerical stability: max diff is {(original_pipe.unet.get_parameter(param_name) - pipe.unet.get_parameter(param_name)).abs().max().item()}"
    )
