"""OneFlow ComfyUI Speedup Module"""
from ._config import _USE_UNET_INT8
from ._nodes import (
    ModelSpeedup,
    ModelGraphLoader,
    ModelGraphSaver,
    VaeSpeedup,
    VaeGraphLoader,
    VaeGraphSaver,
    ControlNetSpeedup,
    ControlNetGraphLoader,
    ControlNetGraphSaver,
    SVDSpeedup,
    ModuleDeepCacheSpeedup,
    OneDiffCheckpointLoaderSimple,
)
from ._compare_node import CompareModel, ShowImageDiff


NODE_CLASS_MAPPINGS = {
    "ModelSpeedup": ModelSpeedup,
    "CompareModel": CompareModel,
    "ShowImageDiff": ShowImageDiff,
    "ModelGraphLoader": ModelGraphLoader,
    "ModelGraphSaver": ModelGraphSaver,
    "VaeSpeedup": VaeSpeedup,
    "VaeGraphSaver": VaeGraphSaver,
    "VaeGraphLoader": VaeGraphLoader,
    "ControlNetSpeedup": ControlNetSpeedup,
    "ControlNetGraphLoader": ControlNetGraphLoader,
    "ControlNetGraphSaver": ControlNetGraphSaver,
    "SVDSpeedup": SVDSpeedup,
    "ModuleDeepCacheSpeedup": ModuleDeepCacheSpeedup,
    "OneDiffCheckpointLoaderSimple": OneDiffCheckpointLoaderSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelSpeedup": "Model Speedup",
    "CompareModel": "Model Weight Comparator",
    "ShowImageDiff": "Image Distinction Scanner",
    "ModelGraphLoader": "Model Graph Loader",
    "ModelGraphSaver": "Model Graph Saver",
    "VaeSpeedup": "VAE Speedup",
    "VaeGraphLoader": "VAE Graph Loader",
    "VaeGraphSaver": "VAE Graph Saver",
    "ControlNetSpeedup": "ControlNet Speedup",
    "ControlNetGraphLoader": "ControlNet Graph Loader",
    "ControlNetGraphSaver": "ControlNet Graph Saver",
    "SVDSpeedup": "SVD Speedup",
    "ModuleDeepCacheSpeedup": "Model DeepCache Speedup",
    "OneDiffCheckpointLoaderSimple": "Load Checkpoint - OneDiff",
}

if _USE_UNET_INT8:
    from ._nodes import UNETLoaderInt8, Quant8Model

    NODE_CLASS_MAPPINGS.update(
        {"UNETLoaderInt8": UNETLoaderInt8, "Quant8Model": Quant8Model}
    )
    NODE_DISPLAY_NAME_MAPPINGS.update(
        {
            "UNETLoaderInt8": "UNET Loader Int8",
            "Quant8Model": "Model Quantization(int8)",
        }
    )
