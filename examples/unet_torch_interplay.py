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


class MockCtx(object):
    def __enter__(self):
        flow.mock_torch.enable(lazy=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        flow.mock_torch.disable()


def get_unet(token):
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        use_auth_token=token,
        revision="fp16",
        torch_dtype=flow.float16,
        subfolder="unet",
    )
    with flow.no_grad():
        unet = unet.to("cuda")
    return unet


class UNetGraphWithCache(flow.nn.Graph):
    @flow.nn.Graph.with_dynamic_input_shape(size=9)
    def __init__(self, unet):
        super().__init__(enable_get_runtime_state_dict=True)
        self.unet = unet
        self.config.enable_cudnn_conv_heuristic_search_algo(False)
        self.config.allow_fuse_add_to_output(True)

    def build(self, latent_model_input, t, text_embeddings):
        text_embeddings = flow._C.amp_white_identity(text_embeddings)
        return self.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

    def warmup_with_arg(self, arg_meta_of_sizes):
        for arg_metas in arg_meta_of_sizes:
            print(f"warmup {arg_metas=}")
            arg_tensors = [flow.empty(a[0], dtype=a[1]).to("cuda") for a in arg_metas]
            self(*arg_tensors)  # build and warmup

    def warmup_with_load(self, file_path):
        state_dict = flow.load(file_path)
        self.load_runtime_state_dict(state_dict)

    def save_graph(self, file_path):
        state_dict = self.runtime_state_dict()
        flow.save(state_dict, file_path)


def image_dim(i):
    return 768 + 128 * i


def noise_shape(batch_size, num_channels, image_w, image_h):
    sizes = (image_w // 8, image_h // 8)
    return (batch_size, num_channels) + sizes


def get_arg_meta_of_sizes(batch_sizes, resolution_scales, num_channels):
    return [
        [
            (
                noise_shape(batch_size, num_channels, image_dim(i), image_dim(j)),
                flow.float16,
            ),
            ((1,), flow.int64),
            ((batch_size, 77, 768), flow.float16),
        ]
        for batch_size in batch_sizes
        for i in resolution_scales
        for j in resolution_scales
    ]


@click.command()
@click.option("--token")
@click.option("--repeat", default=100)
@click.option("--sync_interval", default=50)
@click.option("--save", is_flag=True)
@click.option("--load", is_flag=True)
@click.option("--file", type=str, default="./unet_graphs")
def benchmark(token, repeat, sync_interval, save, load, file):
    RESOLUTION_SCALES = [2, 1, 0]
    BATCH_SIZES = [2]

    # create a mocked unet graph
    num_channels = 4

    warmup_meta_of_sizes = get_arg_meta_of_sizes(BATCH_SIZES, RESOLUTION_SCALES, num_channels)
    for (i, m) in enumerate(warmup_meta_of_sizes):
        print(f"warmup case #{i + 1}:", m)
    with MockCtx():
        unet = get_unet(token)
        unet_graph = UNetGraphWithCache(unet)
        if load == True:
            print("loading graphs...")
            unet_graph.warmup_with_load(file)
        else:
            print("warmup with arguments...")
            unet_graph.warmup_with_arg(warmup_meta_of_sizes)

    # generate inputs with torch
    from diffusers.utils import floats_tensor
    import torch

    time_step = torch.tensor([10]).to("cuda")
    encoder_hidden_states_of_sizes = {
        batch_size: floats_tensor((batch_size, 77, 768)).to("cuda").to(torch.float16)
        for batch_size in BATCH_SIZES
    }
    noise_of_sizes = [
        floats_tensor(noise_shape(batch_size, num_channels, image_dim(i), image_dim(j)))
        .to("cuda")
        .to(torch.float16)
        for batch_size in BATCH_SIZES
        for i in RESOLUTION_SCALES
        for j in RESOLUTION_SCALES
    ]
    noise_of_sizes = [flow.utils.tensor.from_torch(x) for x in noise_of_sizes]
    encoder_hidden_states_of_sizes = {
        k: flow.utils.tensor.from_torch(v) for k, v in encoder_hidden_states_of_sizes.items()
    }
    # convert to oneflow tensors
    time_step = flow.utils.tensor.from_torch(time_step)

    flow._oneflow_internal.eager.Sync()
    import time

    t0 = time.time()
    for r in tqdm(range(repeat)):
        import random

        noise = random.choice(noise_of_sizes)
        encoder_hidden_states = encoder_hidden_states_of_sizes[noise.shape[0]]
        out = unet_graph(noise, time_step, encoder_hidden_states)
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

    if save:
        print("saving graphs...")
        unet_graph.save_graph(file)


if __name__ == "__main__":
    print(f"{flow.__path__=}")
    print(f"{flow.__version__=}")
    benchmark()
