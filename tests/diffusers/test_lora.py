import unittest
import random
from PIL import Image
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import numpy as np
import safetensors.torch
from skimage.metrics import structural_similarity
from diffusers import DiffusionPipeline
from onediff.infer_compiler import oneflow_compile

from onediffx.utils.lora import load_and_fuse_lora, unfuse_lora

MODEL_ID = "/share_nfs/hf_models/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID, variant="fp16", torch_dtype=torch.float16
).to("cuda")
pipe.unet = oneflow_compile(pipe.unet)
pipe(
    "masterpiece, best quality, mountain",
    height=1024,
    width=1024,
    num_inference_steps=30,
).images[0]

loras = [
    "/share_nfs/onediff_ci/diffusers/loras/SDXL-Emoji-Lora-r4.safetensors",
    "/share_nfs/onediff_ci/diffusers/loras/sdxl_metal_lora.safetensors",
    "/share_nfs/onediff_ci/diffusers/loras/simple_drawing_xl_b1-000012.safetensors",
    "/share_nfs/onediff_ci/diffusers/loras/texta.safetensors",
    "/share_nfs/onediff_ci/diffusers/loras/watercolor_v1_sdxl_lora.safetensors",
]
loras = {x: safetensors.torch.load_file(x) for x in loras}
image_file_prefix = "/share_nfs/onediff_ci/diffusers/images"

weight_num = 20
weight_names = {
    "unet": [x for x, _ in pipe.unet.named_parameters()],
    "text_encoder": [x for x, _ in pipe.text_encoder.named_parameters()],
}
if hasattr(pipe, "text_encoder_2"):
    weight_names.update(
        {"text_encoder_2": [x for x, _ in pipe.text_encoder_2.named_parameters()]}
    )
random_weight_names = {k: random.choices(weight_names[k], k=weight_num) for k in weight_names}
original_weights = defaultdict(list)
for part, names in random_weight_names.items():
    for name in names:
        original_weights[part].append([name, getattr(pipe, part).get_parameter(name)])

def check_param(pipe, original_weights: Dict[str, List[Tuple[str, torch.Tensor]]]):
    for part, state_dict in original_weights.items():
        for name, weight in state_dict:
            current_weight = getattr(pipe, part).get_parameter(name)
            return torch.allclose(current_weight, weight)

class TestLoRA(unittest.TestCase):
    def test_lora_loading(test_case):
        for name, lora in loras.items():
            generator = torch.manual_seed(0)
            load_and_fuse_lora(pipe, lora.copy())
            images_fusion = pipe(
                "a cat", generator=generator, height=1024, width=1024, num_inference_steps=30,
            ).images[0]
            image_name = name.split("/")[-1].split(".")[0]
            target_image = np.array(Image.open(f"{image_file_prefix}/test_sdxl_lora_{image_name}.png"))
            curr_image = np.array(images_fusion)
            ssim = structural_similarity(
                curr_image, target_image, channel_axis=-1, data_range=255
            )
            unfuse_lora(pipe)
            assert ssim > 0.98
    
    def test_lora_switching(test_case):
        for lora in loras.values():
            device = random.choice(["cpu", "cuda"])
            weight = random.choice(["lora", "weight"])
            load_and_fuse_lora(
                pipe, lora.copy(), lora_scale=1.0, offload_device=device, offload_weight=weight
            )
            unfuse_lora(pipe)
            assert check_param(pipe, original_weights)

if __name__ == "__main__":
    unittest.main()