import os

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

import click
import oneflow as flow
from tqdm import tqdm


def mock_wrapper(f):
    import sys

    flow.mock_torch.enable(lazy=True)
    ret = f()
    flow.mock_torch.disable()
    # TODO: this trick of py mod purging will be removed
    tmp = sys.modules.copy()
    for x in tmp:
        if x.startswith("diffusers"):
            del sys.modules[x]
    return ret


class UNetGraph(flow.nn.Graph):
    @flow.nn.Graph.with_dynamic_input_shape
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


def get_graph(unet):
    with flow.no_grad():
        unet = unet.to("cuda")
        return UNetGraph(unet)


class UnetCache:
    def __init__(self, token, graph_getter, arg_meta_of_sizes):
        from diffusers import UNet2DConditionModel

        self.unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            use_auth_token=token,
            revision="fp16",
            torch_dtype=flow.float16,
            subfolder="unet",
        )
        self.g = graph_getter(self.unet)
        self.g.debug(0)
        for arg_metas in arg_meta_of_sizes:
            print(f"{arg_metas=}")
            arg_tensors = [flow.empty(a[0], dtype=a[1]).to("cuda") for a in arg_metas]
            self.g(*arg_tensors)  # build and warmup

    def __call__(self, *arg_tensors):
        return self.g(*arg_tensors)


# TODO: noise shape might change by batch
def noise_shape(batch_size, num_channels, image_w, image_h):
    sizes = (image_w // 8, image_h // 8)
    return (batch_size, num_channels) + sizes


test_seq = [2, 1, 0]


def image_dim(i):
    return 768 + 128 * i


@click.command()
@click.option("--token")
@click.option("--repeat", default=1000)
@click.option("--sync_interval", default=50)
def benchmark(token, repeat, sync_interval):
    # create a mocked unet graph
    batch_size = 2
    num_channels = 4

    graph_cache = mock_wrapper(
        lambda: UnetCache(
            token,
            get_graph,
            [
                [
                    (
                        noise_shape(
                            batch_size, num_channels, image_dim(i), image_dim(j)
                        ),
                        flow.float16,
                    ),
                    ((1,), flow.int64),
                    ((batch_size, 77, 768), flow.float16),
                ]
                for i in test_seq
                for j in test_seq
            ],
        )
    )

    # generate inputs with torch
    from diffusers.utils import floats_tensor
    import torch

    sizes = (64, 64)
    noise = (
        floats_tensor((batch_size, num_channels) + sizes).to("cuda").to(torch.float16)
    )
    print(f"{type(noise)=}")
    time_step = torch.tensor([10]).to("cuda")
    encoder_hidden_states = (
        floats_tensor((batch_size, 77, 768)).to("cuda").to(torch.float16)
    )

    noise_of_sizes = [
        floats_tensor(noise_shape(batch_size, num_channels, image_dim(i), image_dim(j)))
        .to("cuda")
        .to(torch.float16)
        for i in test_seq
        for j in test_seq
    ]
    noise_of_sizes = [flow.utils.tensor.from_torch(x) for x in noise_of_sizes]
    # convert to oneflow tensors
    [time_step, encoder_hidden_states] = [
        flow.utils.tensor.from_torch(x) for x in [time_step, encoder_hidden_states]
    ]

    flow._oneflow_internal.eager.Sync()
    import time

    t0 = time.time()
    for r in tqdm(range(repeat)):
        import random

        noise = random.choice(noise_of_sizes)
        out = graph_cache(noise, time_step, encoder_hidden_states)
        # convert to torch tensors
        out = flow.utils.tensor.to_torch(out)
        if r == repeat - 1 or r % sync_interval == 0:
            flow._oneflow_internal.eager.Sync()
    print(f"{type(out)=}")
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
