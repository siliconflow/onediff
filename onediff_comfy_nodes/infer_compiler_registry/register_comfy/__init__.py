import os
from onediff.infer_compiler import register
from nodes import *  # must imported before import comfy
from pathlib import Path

comfy_path = Path(os.path.abspath(__file__)).parents[4] / "comfy"
register(package_names=[comfy_path])
import comfy
from .attention import CrossAttention as CrossAttention1f
from .attention import SpatialTransformer as SpatialTransformer1f
from .linear import Linear as Linear1f
from .deep_cache_unet import DeepCacheUNet
from .deep_cache_unet import FastDeepCacheUNet


if hasattr(comfy.ops, "disable_weight_init"):
    comfy_ops_Linear = comfy.ops.disable_weight_init.Linear
else:
    print(
        "Warning: ComfyUI version is too old, please upgrade it. github: git@github.com:comfyanonymous/ComfyUI.git "
    )
    comfy_ops_Linear = comfy.ops.Linear

torch2of_class_map = {
    comfy.ldm.modules.attention.CrossAttention: CrossAttention1f,
    comfy.ldm.modules.attention.SpatialTransformer: SpatialTransformer1f,
    comfy_ops_Linear: Linear1f,
}


register(torch2oflow_class_map=torch2of_class_map)
