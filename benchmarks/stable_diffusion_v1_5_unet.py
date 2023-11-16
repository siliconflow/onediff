"""
example: python examples/stable_diffusion_v1_5_unet.py --model_id runwayml/stable-diffusion-v1-5
"""
import os
import time
import torch
import click
from tqdm import tqdm
import oneflow as flow
from onediff.infer_compiler import oneflow_compile
from onediff.infer_compiler.utils.set_oneflow_environment import set_oneflow_environment
from diffusers import UNet2DConditionModel
from diffusers.utils import floats_tensor


set_oneflow_environment()

del os.environ["ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT"]


class UNetGraph(flow.nn.Graph):
    """build unet graph"""

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
@click.option("--model_id", default="runwayml/stable-diffusion-v1-5")
def benchmark(token, repeat, sync_interval, model_id):
    with flow.no_grad():
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
        sizes = (64, 64)
        noise = (
            floats_tensor((batch_size, num_channels) + sizes)
            .to("cuda")
            .to(torch.float16)
        )
        time_step = flow.tensor([10]).to("cuda")
        encoder_hidden_states = (
            floats_tensor((batch_size, 77, 768)).to("cuda").to(torch.float16)
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
    print(f"{flow.__path__=}")
    print(f"{flow.__version__=}")
    benchmark()
