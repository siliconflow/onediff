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
        # TODO: this trick of py mod purging will be removed
        import sys
        tmp = sys.modules.copy()
        for x in tmp:
            if x.startswith("diffusers"):
                del sys.modules[x]


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

def get_graph(unet):
    class UNetGraph(flow.nn.Graph):
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

    return UNetGraph(unet)


class GraphUtil(object):
    def __init__(self, graph):
        self._g = graph
        self._g.debug(0)
    
    def warmup_with_arg(self, arg_meta_of_sizes):
        for arg_metas in arg_meta_of_sizes:
            print(f"warmup {arg_metas=}")
            arg_tensors = [flow.empty(a[0], dtype=a[1]).to("cuda") for a in arg_metas]
            self._g(*arg_tensors)  # build and warmup
    
    def warmup_with_load(self, file_path):
        state_dict = flow.load(file_path)
        self._g.load_runtime_state_dict(state_dict)

    def save_graph(self, file_path):
        state_dict = self._g.runtime_state_dict()
        flow.save(state_dict, file_path)


test_seq = [2, 1, 0]

def image_dim(i):
    return 768 + 128 * i

# TODO: noise shape might change by batch
def noise_shape(batch_size, num_channels, image_w, image_h):
    sizes = (image_w // 8, image_h // 8)
    return (batch_size, num_channels) + sizes

def get_arg_meta_of_sizes(batch_size, num_channels):
    ret =  [
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
            ]
    return ret



@click.command()
@click.option("--token")
@click.option("--repeat", default=10)
@click.option("--sync_interval", default=5)
@click.option("--save", type=bool, default=False)
@click.option("--load", type=bool, default=False)
@click.option("--file", type=str, default="./unet_graphs")
def benchmark(token, repeat, sync_interval, save, load, file):
    # create a mocked unet graph
    batch_size = 2
    num_channels = 4
    with MockCtx():
        unet = get_unet(token)
        unet_graph = get_graph(unet)
        unet_graph_util = GraphUtil(unet_graph)
        if load == True :
            print("warmup_with_load")
            unet_graph_util.warmup_with_load(file)
        else:
            print("warmup_with_arg")
            unet_graph_util.warmup_with_arg(get_arg_meta_of_sizes(batch_size, num_channels))
        
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
        out = unet_graph(noise, time_step, encoder_hidden_states)
        # convert to torch tensors
        out = flow.utils.tensor.to_torch(out)
        if r == repeat - 1 or r % sync_interval == 0:
            flow._oneflow_internal.eager.Sync()
    print(f"{type(out)=}")
    flow._oneflow_internal.eager.Sync()
    t1 = time.time()
    duration = t1 - t0
    throughput = repeat / duration
    print(
        f"Finish {repeat} steps in {duration:.3f} seconds, average {throughput:.2f}it/s"
    )

    if save:
        print("save_graph")
        unet_graph_util.save_graph(file)


if __name__ == "__main__":
    print(f"{flow.__path__=}")
    print(f"{flow.__version__=}")
    benchmark()
