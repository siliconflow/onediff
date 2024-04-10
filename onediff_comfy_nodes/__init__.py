"""OneDiff ComfyUI Speedup Module"""
from ._config import *
from ._nodes import (ModelSpeedup, OneDiffApplyModelOptimizer,
                     OneDiffCheckpointLoaderSimple, OneDiffControlNetLoader,
                     VaeSpeedup)

NODE_CLASS_MAPPINGS = {
    "ModelSpeedup": ModelSpeedup,
    "VaeSpeedup": VaeSpeedup,
    "OneDiffModelOptimizer": OneDiffApplyModelOptimizer, 
    "OneDiffCheckpointLoaderSimple": OneDiffCheckpointLoaderSimple,
    "OneDiffControlNetLoader": OneDiffControlNetLoader,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelSpeedup": "Model Speedup",
    "VaeSpeedup": "VAE Speedup",
    "OneDiffModelOptimizer": "Apply Model Optimizer - OneDff",
    "OneDiffCheckpointLoaderSimple": "Load Checkpoint - OneDiff",
}

from .extras_nodes import (nodes_compare, nodes_oneflow_optimizer,
                           nodes_torch_compile_optimizer)

extras = [nodes_compare, nodes_oneflow_optimizer, nodes_torch_compile_optimizer]
for node in extras:
    NODE_CLASS_MAPPINGS.update(node.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(node.NODE_DISPLAY_NAME_MAPPINGS)

