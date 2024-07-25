## Environment setup
### Set up ComfyUI
https://github.com/comfyanonymous/ComfyUI

### Set up onediff_comfy_nodes
https://github.com/siliconflow/onediff?tab=readme-ov-file#installation

```shell
# python 3.10
git clone https://github.com/siliconflow/onediff.git
cd onediff && pip install -e .
ln -s $(pwd)/onediff_comfy_nodes path/to/ComfyUI/custom_nodes/
# or
# cp -r onediff_comfy_nodes path/to/ComfyUI/custom_nodes/
```
### Set up nexfort backend
https://github.com/siliconflow/onediff/tree/main/src/onediff/infer_compiler/backends/nexfort

## Getting Started
### Run ComfyUI
Note ⚠️: Replace 'path/to/' with the actual path to the directories and files on your system.
```shell
export COMFYUI_ROOT=path/to/ComfyUI

cd path/to/onediff/onediff_comfy_nodes/benchmarks

bash scripts/install_env.sh $COMFYUI_ROOT

cd  $COMFYUI_ROOT

python main.py --gpu-only --port 8188 --extra-model-paths-config path/to/onediff/tests/comfyui/extra_model_paths.yaml
```

## Usage Example

```shell
cd path/to/onediff/onediff_comfy_nodes/benchmarks

bash scripts/run_text_to_image.sh
```

The output results will be saved in the results/ directory.

## How to add a workflow for testing
To add a workflow for testing, you can refer to the `run_text_to_image.sh` script and the `input_registration.py` file in the `src` directory. Here's an example of how to register a workflow generator:

```python
# file: onediff/onediff_comfy_nodes/benchmarks/src/input_registration.py

@register_generator(f"{WORKFLOW_DIR}/example_workflow_api.json")
def _(workflow_path, *args, **kwargs):
    with open(workflow_path, "r") as fp:
        workflow = json.load(fp)
    graph = ComfyGraph(graph=workflow, sampler_nodes=["3"])
    for height in [1024, 768, 512]:
        for width in [1024, 768, 512]:
            graph.set_image_size(height=height, width=width)
            yield InputParams(graph=graph)
```
