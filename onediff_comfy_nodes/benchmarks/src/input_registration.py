import json
import os
from typing import NamedTuple
from core.registry import create_generator_registry
from core.service_client import ComfyGraph


class InputParams(NamedTuple):
    graph: ComfyGraph


def read_prompts(file_path="resources/prompts.txt"):
    with open(file_path, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
    return [line.strip() for line in lines if line.strip()]


def get_all_images(
    directory="/share_nfs/hf_models/comfyui_resources/input/faces",
    image_extensions=set([".jpg", ".jpeg", ".png", ".gif", ".bmp"]),
):
    all_files = [os.path.join(directory, f) for f in os.listdir(directory)]
    image_files = [
        f
        for f in all_files
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in image_extensions
    ]
    return image_files


# Create register and get functions
register_generator, dispatch_generator = create_generator_registry()


@register_generator("resources/example_workflow_api.json")
def _(workflow_path, *args, **kwargs):
    with open(workflow_path, "r") as fp:
        workflow = json.load(fp)
    graph = ComfyGraph(graph=workflow, sampler_nodes=["3"])
    for height in [1024, 768, 512]:
        for width in [1024, 768, 512]:
            graph.set_image_size(height=height, width=width)
            yield InputParams(graph=graph)


SD3_WORKFLOWS = [
    "resources/baseline/sd3_baseline.json",
    "resources/nexfort/sd3_unet_speedup.json",
    "resources/nexfort/sd3_unet_vae_speedup.json",
]


@register_generator(SD3_WORKFLOWS)
def _(workflow_path, *args, **kwargs):
    with open(workflow_path, "r") as fp:
        workflow = json.load(fp)

    graph = ComfyGraph(graph=workflow, sampler_nodes=["271"])
    texts = read_prompts()
    for height in [1024, 768, 512]:
        for width in [1024, 768, 512]:
            for text in texts[-5:]:
                graph.set_prompt(prompt=text)
                graph.set_image_size(height=height, width=width)
                yield InputParams(graph=graph)


@register_generator("resources/oneflow/sdxl-control-lora-speedup.json")
def _(workflow_path, *args, **kwargs):
    with open(workflow_path, "r") as fp:
        workflow = json.load(fp)

    graph = ComfyGraph(graph=workflow, sampler_nodes=["1"])
    yield InputParams(graph=graph)


@register_generator(
    [
        "resources/baseline/ComfyUI_IPAdapter_plus/ipadapter_advanced.json",
        "resources/oneflow/ComfyUI_IPAdapter_plus/ipadapter_advanced.json",
    ]
)
def _(workflow_path, *args, **kwargs):
    with open(workflow_path, "r") as fp:
        workflow = json.load(fp)

    graph = ComfyGraph(graph=workflow, sampler_nodes=["3"])
    for image in get_all_images():
        graph.graph["12"]["inputs"]["image"] = image
        for height in [768, 512]:
            for width in [768, 512]:
                positive_prompt = "in a peaceful spring morning a woman wearing a white shirt is sitting in a park on a bench\n\nhigh quality, detailed, diffuse light"
                negative_prompt = "blurry, noisy, messy, lowres, jpeg, artifacts, ill, distorted, malformed"
                graph.set_prompt(positive_prompt, negative_prompt)
                graph.set_image_size(height=height, width=width)
                yield InputParams(graph=graph)


@register_generator(
    [
        "resources/baseline/lora.json",
        "resources/baseline/lora_multiple.json",
        "resources/oneflow/lora_speedup.json",
        "resources/oneflow/lora_multiple_speedup.json",
    ]
)
def _(workflow_path, *args, **kwargs):
    with open(workflow_path, "r") as fp:
        workflow = json.load(fp)

    graph = ComfyGraph(graph=workflow, sampler_nodes=["3"])
    graph.set_prompt("masterpiece best quality girl, hanfu", "bad hands")
    root_path = "sd15/"
    checkpoint_nodes = []
    for node in graph.graph.values():
        if node["class_type"] in [
            "CheckpointLoaderSimple",
            "OneDiffCheckpointLoaderSimple",
        ]:
            checkpoint_nodes.append(node)
    assert len(checkpoint_nodes) == 1

    for file_name in [
        "020.realisticVisionV51_v51VAE.safetensors",
        "v1-5-pruned-emaonly.ckpt",
    ]:
        checkpoint_path = os.path.join(root_path, file_name)
        checkpoint_nodes[0]["inputs"]["ckpt_name"] = checkpoint_path
        yield InputParams(graph=graph)
