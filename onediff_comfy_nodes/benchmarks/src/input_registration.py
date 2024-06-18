import json
from typing import NamedTuple
from core import create_constructor_registry, ComfyGraph, logger


def read_prompts(file_path="resources/prompts.txt"):
    with open(file_path, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
    return [line.strip() for line in lines if line.strip()]


# Create register and get functions
register_constructor, get_input_constructor = create_constructor_registry()


@register_constructor("resources/example_workflow_api.json")
def _(workflow_api_file_path):
    with open(workflow_api_file_path, "r") as fp:
        workflow = json.load(fp)
    graph = ComfyGraph(graph=workflow, sampler_nodes=["3"])
    graph.set_prompt("masterpiece best quality man")
    yield graph
    for height in [1024, 768, 512]:
        for width in [1024, 768, 512]:
            logger.info(f"ComfyGraph: {height=} {width=}")
            graph.set_image_size(height=height, width=width)
            yield graph


SD3_WORKFLOWS = [
    "resources/baseline/sd3_baseline.json",
    "resources/nexfort/sd3_unet_speedup.json",
    "resources/nexfort/sd3_unet_vae_speedup.json",
]


@register_constructor(SD3_WORKFLOWS)
def _(workflow_api_file_path):
    with open(workflow_api_file_path, "r") as fp:
        workflow = json.load(fp)

    graph = ComfyGraph(graph=workflow, sampler_nodes=["271"])
    texts = read_prompts()
    for height in [1024, 768, 512]:
        for width in [1024, 768, 512]:
            for text in texts[-5:]:
                graph.set_prompt(prompt=text)
                logger.info(f"ComfyGraph: {height=} {width=}")
                graph.set_image_size(height=height, width=width)
                yield graph
