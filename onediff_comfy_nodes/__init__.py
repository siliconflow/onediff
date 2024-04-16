"""OneDiff ComfyUI Speedup Module"""
from ._config import *
from ._nodes import (ModelSpeedup, OneDiffApplyModelBooster,
                     OneDiffCheckpointLoaderSimple, OneDiffControlNetLoader,
                     VaeSpeedup)

NODE_CLASS_MAPPINGS = {
    "ModelSpeedup": ModelSpeedup,
    "VaeSpeedup": VaeSpeedup,
    "OneDiffModelBooster": OneDiffApplyModelBooster, 
    "OneDiffCheckpointLoaderSimple": OneDiffCheckpointLoaderSimple,
    "OneDiffControlNetLoader": OneDiffControlNetLoader,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelSpeedup": "Model Speedup",
    "VaeSpeedup": "VAE Speedup",
    "OneDiffModelBooster": "Apply Model Booster - OneDff",
    "OneDiffCheckpointLoaderSimple": "Load Checkpoint - OneDiff",
}

from .extras_nodes import (nodes_compare, nodes_oneflow_booster, nodes_torch_compile_booster)

extras = [nodes_compare, nodes_oneflow_booster, nodes_torch_compile_booster]
for node in extras:
    NODE_CLASS_MAPPINGS.update(node.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(node.NODE_DISPLAY_NAME_MAPPINGS)

