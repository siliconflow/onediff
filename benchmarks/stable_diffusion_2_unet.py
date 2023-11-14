
"""
test run graph
"""

import os
import time
import click
from tqdm import tqdm
import torch
import oneflow as flow
from onediff.infer_compiler import oneflow_compile
from diffusers import UNet2DConditionModel
from diffusers.utils import floats_tensor

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


@click.command()
@click.option("--token")
@click.option("--height", default=768)
@click.option("--width", default=768)
@click.option("--repeat", default=1000)
@click.option("--sync_interval", default=50)
@click.option("--model_id", default="stabilityai/stable-diffusion-2")
def benchmark(token, height, width, repeat, sync_interval, model_id):
    """Function test use graph speed."""
    with torch.no_grad():
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            use_auth_token=token,
            revision="fp16",
            torch_dtype=torch.float16,
            subfolder="unet",
        )
        unet = unet.to("cuda")
        unet_graph = oneflow_compile(unet)

        batch_size = 2
        num_channels = 4
        sizes = (height // 8, width // 8)
        noise = (
            floats_tensor((batch_size, num_channels) + sizes)
            .to("cuda")
            .to(torch.float16)
        )
        time_step = torch.tensor([10]).to("cuda")
        encoder_hidden_states = (
            floats_tensor((batch_size, 77, 1024)).to("cuda").to(torch.float16)
        )
        unet_graph(noise, time_step, encoder_hidden_states)
        flow._oneflow_internal.eager.Sync()

        t0 = time.time()
        for r in tqdm(range(repeat)):
            unet_graph(noise, time_step, encoder_hidden_states)
            if r == repeat - 1 or r % sync_interval == 0:
                flow._oneflow_internal.eager.Sync()
        t1 = time.time()
        duration = t1 - t0
        throughput = repeat / duration
        print(
            f"Finish {repeat} steps in {duration:.3f} seconds, average {throughput:.2f}it/s"
        )


if __name__ == "__main__":
    benchmark()
