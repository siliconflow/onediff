# HF_HUB_OFFLINE=1 python3 examples/torch_interpretor.py
import os
import torch
from diffusers import StableDiffusionPipeline
from onediff.infer_compiler import torchbackend

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
)

os.environ["with_interp"] = "0"

# optimize with oneflow graph
os.environ["with_graph"] = "1"
os.environ["ONEFLOW_MLIR_CSE"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_OPS_WITH_BACKWARD_IMPL"] = "1"
os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"
os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS"] = "1"
os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"
os.environ["ONEFLOW_KERNEL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"
os.environ["ONEFLOW_KERNEL_CONV_ENABLE_CUTLASS_IMPL"] = "1"
os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
os.environ["ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT"] = "1"


pipe.unet = torch.compile(
    pipe.unet, fullgraph=True, mode="reduce-overhead", backend=torchbackend
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with torch.autocast("cuda"):
    for i in range(3):
        images = pipe(prompt).images
        for j, image in enumerate(images):
            image.save(f"{prompt}-of-{i}-{j}.png")
