import argparse
import os
import time
import torch
import torch.nn as nn
from torch._dynamo import allow_in_graph as maybe_allow_in_graph
import oneflow as flow

# oneflow_compile should be imported before importing any diffusers
from onediff.infer_compiler import oneflow_compile
from onediff.schedulers import EulerDiscreteScheduler
from onediff.optimization import rewrite_self_attention

from diffusers import StableDiffusionXLPipeline

try:
    import onediff_quant
except:
    print("Skip quantized SDXL since onediff_quant is not installed.")
    exit()
from onediff_quant.utils import replace_sub_module_with_quantizable_module

onediff_quant.enable_load_quantized_model()


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument(
    "--prompt",
    type=str,
    default="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--bits", type=int, default=8)
parser.add_argument(
    "--static", default=False, type=(lambda x: str(x).lower() in ["true", "1", "yes"])
)
parser.add_argument(
    "--fake_quant",
    default=False,
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
)
parser.add_argument(
    "--graph",
    default=True,
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--warmup", type=int, default=1)

args = parser.parse_args()

calibrate_info = {}
with open(os.path.join(args.model, "calibrate_info.txt"), "r") as f:
    for line in f.readlines():
        line = line.strip()
        items = line.split(" ")
        calibrate_info[items[0]] = [
            float(items[1]),
            int(items[2]),
            [float(x) for x in items[3].split(",")],
        ]

scheduler = EulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")
pipe = StableDiffusionXLPipeline.from_pretrained(
    args.model, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True
)
pipe.to("cuda")

for sub_module_name, sub_calibrate_info in calibrate_info.items():
    replace_sub_module_with_quantizable_module(
        pipe.unet,
        sub_module_name,
        sub_calibrate_info,
        args.fake_quant,
        args.static,
        args.bits,
        maybe_allow_in_graph,
    )

# if pipe.text_encoder is not None:
#     pipe.text_encoder = oneflow_compile(pipe.text_encoder, use_graph=args.graph)
# pipe.text_encoder_2 = oneflow_compile(pipe.text_encoder_2, use_graph=args.graph)
if args.graph:
    rewrite_self_attention(pipe.unet)
pipe.unet = oneflow_compile(pipe.unet, use_graph=args.graph)
pipe.vae.decoder = oneflow_compile(pipe.vae.decoder, use_graph=args.graph)

for _ in range(args.warmup):
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
    ).images[0]

torch.manual_seed(args.seed)

start_t = time.time()

image = pipe(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.steps,
).images[0]

end_t = time.time()
cuda_memory_usage = flow._oneflow_internal.GetCUDAMemoryUsed()
print(
    f"e2e ({args.steps} steps) elapsed: {end_t - start_t} s, cuda memory usage: {cuda_memory_usage} MiB"
)
