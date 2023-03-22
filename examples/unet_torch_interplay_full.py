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
from unet_torch_interplay import image_dim, noise_shape

class MockCtx(object):
    def __enter__(self):
        flow.mock_torch.enable(lazy=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        flow.mock_torch.disable()


def get_unet(token, _model_id):
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        _model_id,
        use_auth_token=token,
        revision="fp16",
        torch_dtype=flow.float16,
        subfolder="unet",
    )
    with flow.no_grad():
        unet = unet.to("cuda")
    return unet

# Get state dict tensors total size in MB
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
            continue
    return tensor_size / 1024 / 1024, string_size / 1024 / 1024

class UNetGraphWithCache(flow.nn.Graph):
    @flow.nn.Graph.with_dynamic_input_shape(size=16)
    def __init__(self, unet):
        super().__init__(enable_get_runtime_state_dict=True)
        self.unet = unet
        self.config.enable_cudnn_conv_heuristic_search_algo(False)
        self.config.allow_fuse_add_to_output(True)

    def build(self, latent_model_input, t, text_embeddings):
        text_embeddings = flow._C.amp_white_identity(text_embeddings)
        # import pdb; pdb.set_trace()
        return self.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

    def warmup_with_arg(self, arg_meta_of_sizes):
        for arg_metas in arg_meta_of_sizes:
            print(f"warmup {arg_metas=}")
            arg_tensors = [flow.empty(a[0], dtype=a[1]).to("cuda") for a in arg_metas]
            self(*arg_tensors)  # build and warmup

    def warmup_with_load(self, file_path):
        import time
        flow._oneflow_internal.eager.Sync()
        t0 = time.time()
        # load state dict from file
        state_dict = flow.load(file_path)
        flow._oneflow_internal.eager.Sync()
        t1 = time.time()
        print(f"load state dict time: {t1 - t0:.3f} seconds")
        print(
            f"state_dict tensors size {_get_state_dict_tensor_size(state_dict)[0]:.3f} MB."
        )
        print(
            f"state_dict string size {_get_state_dict_tensor_size(state_dict)[1]:.3f} MB."
        )
        flow._oneflow_internal.eager.Sync()
        t1 = time.time()
        # load state dict into graph
        self.load_runtime_state_dict(state_dict)
        flow._oneflow_internal.eager.Sync()
        t2 = time.time()
        print(f"load into graph time: {t2 - t1:.3f} seconds")

    def save_graph(self, file_path, with_eager=True):
        import time
        flow._oneflow_internal.eager.Sync()
        t0 = time.time()
        # get state dict from graph
        state_dict = self.runtime_state_dict(with_eager=with_eager)
        flow._oneflow_internal.eager.Sync()
        t1 = time.time()
        print(f"get state dict time: {t1 - t0:.3f} seconds")
        print(
            f"state_dict(with_eager={with_eager}) tensors size {_get_state_dict_tensor_size(state_dict)[0]:.3f} MB"
        )
        print(
            f"state_dict(with_eager={with_eager}) string size {_get_state_dict_tensor_size(state_dict)[1]:.3f} MB"
        )
        flow._oneflow_internal.eager.Sync()
        t1 = time.time()
        # save state dict to file
        flow.save(state_dict, file_path)
        flow._oneflow_internal.eager.Sync()
        t2 = time.time()
        print(f"save state dict time: {t2 - t1:.3f} seconds")


def get_arg_meta_of_sizes(batch_sizes, resolution_scales, num_channels, cross_attention_dim):
    return [
        [
            (
                noise_shape(batch_size, num_channels, image_dim(i), image_dim(j)),
                flow.float16,
            ),
            ((1,), flow.int64),
            # max_length of tokenizer, cross_attention_dim
            ((batch_size, 77, cross_attention_dim), flow.float16),
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
@click.option("--with_eager", is_flag=True)
@click.option("--load", is_flag=True)
@click.option("--file", type=str, default="./unet_graphs")
@click.option("--model_id", type=str, default="stabilityai/stable-diffusion-2")
def benchmark(token, repeat, sync_interval, save, with_eager, load, file, model_id):
    RESOLUTION_SCALES = [3, 2, 1, 0]
    BATCH_SIZES = [2]

    # create a mocked unet graph
    num_channels = 4
    print(f"Model ID: {model_id}")
    with MockCtx():
        unet = get_unet(token, model_id)
        cross_attention_dim = unet.config['cross_attention_dim']
        warmup_meta_of_sizes = get_arg_meta_of_sizes(BATCH_SIZES, RESOLUTION_SCALES, 
                                                     num_channels, cross_attention_dim)
        for (i, m) in enumerate(warmup_meta_of_sizes):
            print(f"warmup case #{i + 1}:", m)
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
        batch_size: floats_tensor((batch_size, 77, cross_attention_dim)).to("cuda").to(torch.float16)
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
        unet_graph.save_graph(file, with_eager)


if __name__ == "__main__":
    print(f"{flow.__path__=}")
    print(f"{flow.__version__=}")
    benchmark()
    