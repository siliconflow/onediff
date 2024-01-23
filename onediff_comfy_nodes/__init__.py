"""OneDiff ComfyUI Speedup Module"""
from ._config import _USE_UNET_INT8
from ._nodes import (
    ModelSpeedup,
    ModelGraphLoader,
    ModelGraphSaver,
    VaeSpeedup,
    VaeGraphLoader,
    VaeGraphSaver,
    ControlNetSpeedup,
    SVDSpeedup,
    ModuleDeepCacheSpeedup,
    OneDiffCheckpointLoaderSimple,
    OneDiffControlNetLoader,
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
    "SVDSpeedup": SVDSpeedup,
    "ModuleDeepCacheSpeedup": ModuleDeepCacheSpeedup,
    "OneDiffCheckpointLoaderSimple": OneDiffCheckpointLoaderSimple,
    "OneDiffControlNetLoader": OneDiffControlNetLoader,
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
    "SVDSpeedup": "SVD Speedup",
    "ModuleDeepCacheSpeedup": "Model DeepCache Speedup",
    "OneDiffCheckpointLoaderSimple": "Load Checkpoint - OneDiff",
    "OneDiffControlNetLoader": "Load ControlNet Model - OneDiff",
}


if _USE_UNET_INT8:
    from ._nodes import UNETLoaderInt8, Quant8Model
    from ._nodes import (
        QuantKSampler,
        OneDiffQuantCheckpointLoaderSimple,
        OneDiffQuantCheckpointLoaderSimpleAdvanced,
    )

    NODE_CLASS_MAPPINGS.update(
        {
            "UNETLoaderInt8": UNETLoaderInt8,
            "Quant8Model": Quant8Model,
            "OneDiffQuantCheckpointLoaderSimple": OneDiffQuantCheckpointLoaderSimple,
            "OneDiffQuantCheckpointLoaderSimpleAdvanced": OneDiffQuantCheckpointLoaderSimpleAdvanced,
            "QuantKSampler": QuantKSampler,
        }
    )

    NODE_DISPLAY_NAME_MAPPINGS.update(
        {
            "UNETLoaderInt8": "UNET Loader Int8",
            "Quant8Model": "Model Quantization(int8)",
            "OneDiffQuantCheckpointLoaderSimple": "Load Checkpoint - OneDiff Quant",
            "OneDiffQuantCheckpointLoaderSimpleAdvanced": "Load Checkpoint - OneDiff Quant Advanced",
            "QuantKSampler": "Quant K Sampler",
        }
    )
