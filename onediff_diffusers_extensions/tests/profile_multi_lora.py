import time
from pathlib import Path

import pandas as pd
import safetensors.torch
import torch
from diffusers import DiffusionPipeline
from diffusers.utils.constants import USE_PEFT_BACKEND

from onediff.infer_compiler import oneflow_compile
from onediff.torch_utils import TensorInplaceAssign
from onediffx.lora import load_and_fuse_lora, set_and_fuse_adapters, unfuse_lora

if not USE_PEFT_BACKEND:
    raise RuntimeError(
        "The profile if for PEFT APIs, please make sure you have installed peft>=0.6.0 and transformers >= 4.34.0"
    )


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

# for OneDiffX APIs
for i, (name, lora) in enumerate(loras.items()):
    load_and_fuse_lora(
        pipe,
        lora.copy(),
        adapter_name=Path(name).stem,
        lora_scale=1.0,
        offload_device="cuda",
    )
    unfuse_lora(pipe)

multi_loras = []
for lora in loras:
    lora = Path(lora).stem
    if len(multi_loras) == 0:
        multi_loras.append([lora])
    else:
        multi_loras.append(multi_loras[-1] + [lora])

set_adapter_time = []
for i, multi_lora in enumerate(multi_loras):
    with TimerContextManager("set_adapter", multi_lora):
        set_and_fuse_adapters(pipe, multi_lora, [0.5] * len(multi_lora))
        unfuse_lora(pipe)
    set_adapter_time.append(_time)

# for PEFT APIs
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID, variant="fp16", torch_dtype=torch.float16
).to("cuda")
for i, (name, lora) in enumerate(loras.items()):
    pipe.load_lora_weights(lora.copy(), adapter_name=Path(name).stem, lora_scale=1.0)

peft_set_adapter_time = []
for i, multi_lora in enumerate(multi_loras):
    with TimerContextManager("peft set_adapter", multi_lora):
        pipe.set_adapters(multi_lora, [0.5] * len(multi_lora))
        pipe.fuse_lora(adapter_names=multi_lora)
        pipe.unfuse_lora()
    peft_set_adapter_time.append(_time)

lora_names = [
    [1],
    [1, 2],
    [1, 2, 3],
    [1, 2, 3, 4],
    [1, 2, 3, 4, 5],
]

data = {
    "LoRA names": lora_names,
    "PEFT set_adapter": [f"{x:.2f} s" for x in peft_set_adapter_time],
    "OneDiffX set_adapter": [f"{x:.2f} s" for x in set_adapter_time],
}
df = pd.DataFrame(data)
print(df)

with open("result.md", "w") as file:
    file.write(df.to_markdown(index=False))
