from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from diffusers import DiffusionPipeline
from onediff.infer_compiler import oneflow_compile
from diffusers_extensions.utils.lora import load_and_fuse_lora, unfuse_lora
import safetensors.torch
import random


def check_param(pipe, original_weights: Dict[str, List[Tuple[str, torch.Tensor]]]):
    for part, state_dict in original_weights.items():
        for name, weight in state_dict:
            current_weight = getattr(pipe, part).get_parameter(name)
            if not torch.allclose(current_weight, weight):
                raise ValueError(f"got wrong result of part {part} param name {name}")


MODEL_ID = "/share_nfs/hf_models/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID, variant="fp16", torch_dtype=torch.float16
).to("cuda")

LORA_MODEL_ID = [
    "/data/home/wangyi/models/lora/simple_drawing_xl_b1-000012.safetensors",
    "/data/home/wangyi/models/lora/texta.safetensors",
    "/data/home/wangyi/workspace/stable-diffusion-webui/models/Lora/watercolor_v1_sdxl_lora.safetensors",
    "/data/home/wangyi/workspace/stable-diffusion-webui/models/Lora/SDXL-Emoji-Lora-r4.safetensors",
    "/data/home/wangyi/workspace/stable-diffusion-webui/models/Lora/sdxl_metal_lora.safetensors",
]
LORA_MODEL_ID = list(map(lambda x: safetensors.torch.load_file(x), LORA_MODEL_ID))
pipe.unet = oneflow_compile(pipe.unet)
generator = torch.manual_seed(0)

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

for lora in LORA_MODEL_ID:
    device = random.choice(["cpu", "cuda"])
    weight = random.choice(["lora", "weight"])
    load_and_fuse_lora(
        pipe, lora.copy(), lora_scale=1.0, offload_device=device, offload_weight=weight
    )
    unfuse_lora(pipe)
    check_param(pipe, original_weights)
