import os

os.environ["ONEFLOW_MLIR_CSE"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"

os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS"] = "1"
os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"

os.environ["ONEFLOW_KERNEL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"
os.environ["ONEFLOW_KERNEL_CONV_ENABLE_CUTLASS_IMPL"] = "1"

os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"

import click
import oneflow as flow
flow.mock_torch.enable()
from diffusers import UNet2DConditionModel
from diffusers.utils import floats_tensor
from tqdm import tqdm


class UNetGraph(flow.nn.Graph):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.config.enable_cudnn_conv_heuristic_search_algo(False)
        self.config.allow_fuse_add_to_output(True)

    def build(self, latent_model_input, t, text_embeddings):
        text_embeddings = flow._C.amp_white_identity(text_embeddings)
        return self.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample


@click.command()
@click.option("--token")
@click.option("--repeat", default=1000)
@click.option("--sync_interval", default=50)
def benchmark(token, repeat, sync_interval):
    with flow.no_grad():
        unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            use_auth_token=token,
            revision="fp16",
            flow_dtype=flow.float16,
            subfolder="unet",
        )
        unet = unet.to("cuda")
        unet_graph = UNetGraph(unet)

        batch_size = 2
        num_channels = 4
        sizes = (64, 64)
        noise = (
            floats_tensor((batch_size, num_channels) + sizes)
            .to("cuda")
            .to(flow.float16)
        )
        time_step = flow.tensor([10]).to("cuda")
        encoder_hidden_states = (
            floats_tensor((batch_size, 77, 768)).to("cuda").to(flow.float16)
        )
        unet_graph(noise, time_step, encoder_hidden_states)
        flow._oneflow_internal.eager.Sync()
        import time

        t0 = time.time()
        for r in tqdm(range(repeat)):
            out = unet_graph(noise, time_step, encoder_hidden_states)
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
