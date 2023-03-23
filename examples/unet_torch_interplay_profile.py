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
import time
from unet_torch_interplay import (
    MockCtx,
    get_unet, 
    get_arg_meta_of_sizes,
    UNetGraphWithCache
)

# Get state dict tensors and string total size in MB
def _get_state_dict_tensor_size(sd):
    from oneflow.framework.args_tree import ArgsTree
    def _get_tensor_mem(input):
        cnt_size = input.element_size() * flow.numel(input)
        return cnt_size

    args_tree = ArgsTree(sd, False)

    tensor_size, string_size = 0, 0
    for arg in args_tree.iter_nodes():
        if isinstance(arg, flow.Tensor):
            tensor_size += _get_tensor_mem(arg)
        elif isinstance(arg, str):
            string_size += len(arg.encode())
        else:
            continue
    return tensor_size / 1024 / 1024, string_size / 1024 / 1024

def _cost_cnt(f):
    def wrapper(*args, **kwargs):
        flow._oneflow_internal.eager.Sync()
        t0 = time.time()
        state_dict = f(*args, **kwargs)
        flow._oneflow_internal.eager.Sync()
        t1 = time.time()
        tensor_size, str_size = _get_state_dict_tensor_size(state_dict)
        print(f"Time cost: {t1 - t0:.3f} seconds")
        print(
            f"state_dict tensors size {tensor_size:.3f} MB; string size {str_size:.3f} MB"
        )
    return wrapper

class UNetGraphWithCacheProfile(UNetGraphWithCache):
    @flow.nn.Graph.with_dynamic_input_shape(size=16)
    def __init__(self, unet):
        super().__init__(unet=unet)

    @_cost_cnt
    def warmup_with_load_profile(self, file_path):
        return self.warmup_with_load(file_path)

    @_cost_cnt
    def save_graph_profile(self, file_path, with_eager=False):
        return self.save_graph(file_path, with_eager=False)

@click.command()
@click.option("--token")
@click.option("--repeat", default=100)
@click.option("--sync_interval", default=50)
@click.option("--save", is_flag=True)
@click.option("--with_eager", is_flag=True)
@click.option("--load", is_flag=True)
@click.option("--file", type=str, default="./unet_graphs")
@click.option("--model_id", type=str, default="stabilityai/stable-diffusion-2")
def benchmark(token, repeat, sync_interval, save, with_eager, load, file, model_id):
    RESOLUTION_SCALES = [3, 2, 1, 0]
    BATCH_SIZES = [2]
    
    num_channels = 4
    print(f"Model ID: {model_id}")
    # create a mocked unet graph
    with MockCtx():
        unet = get_unet(token, model_id)
        unet_graph = UNetGraphWithCacheProfile(unet)
        cross_attention_dim = unet.config['cross_attention_dim']
        warmup_meta_of_sizes = get_arg_meta_of_sizes(BATCH_SIZES, RESOLUTION_SCALES, cross_attention_dim,
                                                     num_channels=num_channels, start=256, stride=256)
        for (i, m) in enumerate(warmup_meta_of_sizes):
            print(f"warmup case #{i + 1}:", m)
        if load == True:
            print("loading graphs...")
            unet_graph.warmup_with_load_profile(file)
        else:
            print("warmup with arguments...")
            unet_graph.warmup_with_arg(warmup_meta_of_sizes)

    # generate inputs with torch
    from diffusers.utils import floats_tensor
    import torch

    time_step = torch.tensor([10]).to("cuda")
    encoder_hidden_states_of_sizes = {
        batch_size: floats_tensor((batch_size, 77, cross_attention_dim)).to("cuda").to(torch.float16)
        for batch_size in BATCH_SIZES
    }
    noise_of_sizes = [
        floats_tensor(warmup_meta_of_sizes[i][0][0])
        .to("cuda")
        .to(torch.float16)
        for i in range(len(warmup_meta_of_sizes))
    ]
    noise_of_sizes = [flow.utils.tensor.from_torch(x) for x in noise_of_sizes]
    encoder_hidden_states_of_sizes = {
        k: flow.utils.tensor.from_torch(v) for k, v in encoder_hidden_states_of_sizes.items()
    }
    # convert to oneflow tensors
    time_step = flow.utils.tensor.from_torch(time_step)

    flow._oneflow_internal.eager.Sync()
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
        f"Finish {repeat} steps in {duration:.3f} seconds, average {throughput:.2f} it/s"
    )

    if save:
        print("saving graphs...")
        unet_graph.save_graph_profile(file, with_eager)


if __name__ == "__main__":
    print(f"{flow.__path__=}")
    print(f"{flow.__version__=}")
    benchmark()
