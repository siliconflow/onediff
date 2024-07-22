"""
Testing inference speed
save graph compiled example: python3 examples/unet_torch_interplay.py --save --model_id xx
load graph compiled example: python3 examples/unet_torch_interplay.py --load
"""
import importlib.metadata
import os
import random

import click

import torch
from packaging import version
import oneflow as flow  # usort: skip

from dataclasses import dataclass, fields

from onediff.infer_compiler import oneflow_compile
from tqdm import tqdm


@dataclass
class TensorInput(object):
    noise: torch.float16
    time: torch.int64
    cross_attention_dim: torch.float16

    @classmethod
    def gettype(cls, key):
        field_types = {field.name: field.type for field in fields(TensorInput)}
        return field_types[key]


def get_unet(token, _model_id, variant):
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        _model_id,
        use_auth_token=token,
        variant=variant,
        torch_dtype=torch.float16,
        subfolder="unet",
    )
    with torch.no_grad():
        unet = unet.to("cuda")
    return unet


def warmup_with_arg(graph, arg_meta_of_sizes, added):
    for arg_metas in arg_meta_of_sizes:
        print(f"warmup {arg_metas=}")
        arg_tensors = [
            torch.empty(arg_metas.noise, dtype=arg_metas.gettype("noise")).to("cuda"),
            torch.empty(arg_metas.time, dtype=arg_metas.gettype("time")).to("cuda"),
            torch.empty(
                arg_metas.cross_attention_dim,
                dtype=arg_metas.gettype("cross_attention_dim"),
            ).to("cuda"),
        ]
        graph(
            *arg_tensors, added_cond_kwargs=added, return_dict=False
        )  # build and warmup


def img_dim(i, start, stride):
    return start + stride * i


def noise_shape(batch_size, num_channels, image_w, image_h):
    sizes = (image_w // 8, image_h // 8)
    return (batch_size, num_channels) + sizes


def get_arg_meta_of_sizes(
    batch_sizes,
    resolution_scales,
    num_channels,
    cross_attention_dim,
    start=768,
    stride=128,
):
    return [
        TensorInput(
            noise_shape(
                batch_size,
                num_channels,
                img_dim(i, start, stride),
                img_dim(j, start, stride),
            ),
            (1,),
            (batch_size, 77, cross_attention_dim),
        )
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
@click.option(
    "--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
)
@click.option("--variant", type=str, default="fp16")
def benchmark(token, repeat, sync_interval, save, load, file, model_id, variant):
    RESOLUTION_SCALES = [2, 1, 0]
    BATCH_SIZES = [2]
    # TODO: reproduce bug caused by changing batch
    # BATCH_SIZES = [4, 2]

    unet = get_unet(token, model_id, variant)
    unet_graph = oneflow_compile(unet)

    num_channels = 4
    cross_attention_dim = unet.config["cross_attention_dim"]
    diffusers_version = version.parse(importlib.metadata.version("diffusers"))
    if diffusers_version < version.parse("0.21.0"):
        from diffusers.utils import floats_tensor
    else:
        from diffusers.utils.testing_utils import floats_tensor

    if (
        model_id == "stabilityai/stable-diffusion-xl-base-1.0"
        or "xl-base-1.0" in model_id
    ):
        # sdxl needed
        add_text_embeds = floats_tensor((2, 1280)).to("cuda").to(torch.float16)
        add_time_ids = floats_tensor((2, 6)).to("cuda").to(torch.float16)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    else:
        added_cond_kwargs = None

    warmup_meta_of_sizes = get_arg_meta_of_sizes(
        batch_sizes=BATCH_SIZES,
        resolution_scales=RESOLUTION_SCALES,
        num_channels=num_channels,
        cross_attention_dim=cross_attention_dim,
    )
    for i, m in enumerate(warmup_meta_of_sizes):
        print(f"warmup case #{i + 1}:", m)

    # load graph from filepath
    if load:
        print("loading graphs...")
        unet_graph.load_graph(file)
    else:
        print("warmup with arguments...")
        warmup_with_arg(unet_graph, warmup_meta_of_sizes, added_cond_kwargs)

    # generate inputs with torch
    time_step = torch.tensor([10]).to("cuda")
    encoder_hidden_states_of_sizes = {
        batch_size: floats_tensor((batch_size, 77, cross_attention_dim))
        .to("cuda")
        .to(torch.float16)
        for batch_size in BATCH_SIZES
    }
    noise_of_sizes = [
        floats_tensor(arg_metas.noise).to("cuda").to(torch.float16)
        for arg_metas in warmup_meta_of_sizes
    ]
    flow._oneflow_internal.eager.Sync()
    import time

    # Testing inference speed
    t0 = time.time()
    for r in tqdm(range(repeat)):
        noise = random.choice(noise_of_sizes)
        encoder_hidden_states = encoder_hidden_states_of_sizes[noise.shape[0]]
        out = unet_graph(
            noise,
            time_step,
            encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        if r == repeat - 1 or r % sync_interval == 0:
            flow._oneflow_internal.eager.Sync()
    assert isinstance(out[0], torch.Tensor)
    print(f"{type(out[0])=}")
    t1 = time.time()
    duration = t1 - t0
    throughput = repeat / duration
    print(
        f"Finish {repeat} steps in {duration:.3f} seconds, average {throughput:.2f}it/s"
    )

    # save graph to filepath
    if save:
        print("saving graphs...")
        unet_graph.save_graph(file)


if __name__ == "__main__":
    print(f"{flow.__path__=}")
    print(f"{flow.__version__=}")
    benchmark()
