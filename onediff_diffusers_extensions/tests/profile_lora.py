import time
from pathlib import Path

import pandas as pd
import safetensors.torch

import torch
from diffusers import DiffusionPipeline

from onediff.infer_compiler import oneflow_compile
from onediff.torch_utils import TensorInplaceAssign
from onediffx.lora import load_and_fuse_lora, unfuse_lora

_time = None


class TimerContextManager:
    def __init__(self, msg, lora):
        self.msg = msg
        self.lora_name = lora

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time - self.start_time
        global _time
        _time = elapsed_time
        print(
            f"Time cost {elapsed_time:.2f}, of method {self.msg}, lora name {self.lora_name}"
        )


MODEL_ID = "/share_nfs/hf_models/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID, variant="fp16", torch_dtype=torch.float16
).to("cuda")

LORA_MODEL_ID = [
    "/share_nfs/onediff_ci/diffusers/loras/SDXL-Emoji-Lora-r4.safetensors",
    "/share_nfs/onediff_ci/diffusers/loras/sdxl_metal_lora.safetensors",
    "/share_nfs/onediff_ci/diffusers/loras/simple_drawing_xl_b1-000012.safetensors",
    "/share_nfs/onediff_ci/diffusers/loras/texta.safetensors",
    "/share_nfs/onediff_ci/diffusers/loras/watercolor_v1_sdxl_lora.safetensors",
]

loras = {name: safetensors.torch.load_file(name) for name in LORA_MODEL_ID}

pipe.unet = oneflow_compile(pipe.unet)
generator = torch.manual_seed(0)

load_lora_weights_time = []
for i, (name, lora) in enumerate(loras.items()):
    with TimerContextManager("load_lora_weights", Path(name).stem):
        pipe.load_lora_weights(lora.copy())
    load_lora_weights_time.append(_time)

modules = [pipe.unet, pipe.text_encoder]
if hasattr(pipe, "text_encoder_2"):
    modules.append(pipe.text_encoder_2)

print("")
load_lora_weights_and_fuse_lora_time = []
for i, (name, lora) in enumerate(loras.items()):
    with TimerContextManager("load_lora_weights and fuse_lora", Path(name).stem):
        pipe.load_lora_weights(lora.copy())
        with TensorInplaceAssign(*modules):
            pipe.fuse_lora(lora_scale=1.0)
            pipe.unfuse_lora()
    load_lora_weights_and_fuse_lora_time.append(_time)

print("")
load_and_fuse_lora_time = []
for i, (name, lora) in enumerate(loras.items()):
    with TimerContextManager("load_and_fuse_lora", Path(name).stem):
        load_and_fuse_lora(
            pipe, lora.copy(), adapter_name=Path(name).stem, lora_scale=1.0
        )
        unfuse_lora(pipe)
    load_and_fuse_lora_time.append(_time)

data = {
    "LoRA name": [Path(x).stem for x in loras],
    "size": ["28M", "23M", "55M", "270M", "12M"],
    "load_lora_weights": [f"{x:.2f} s" for x in load_lora_weights_time],
    "load_lora_weights + fuse_lora": [
        f"{x:.2f} s" for x in load_lora_weights_and_fuse_lora_time
    ],
    "onediffx load_and_fuse_lora": [f"{x:.2f} s" for x in load_and_fuse_lora_time],
    "src link": [
        "[Link](https://novita.ai/model/SDXL-Emoji-Lora-r4_160282)",
        "",
        "[Link](https://civitai.com/models/177820/sdxl-simple-drawing)",
        "[Link](https://civitai.com/models/221240/texta-generate-text-with-sdxl)",
        "",
    ],
}

df = pd.DataFrame(data)
print(df)
# with open("result.md", "w") as file:
#     file.write(df.to_markdown(index=False))
