from onediff.infer_compiler import register
from onediff.infer_compiler.utils import is_community_version
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

if not is_community_version():
    from .openaimodel import Upsample as Upsample1f
    from .openaimodel import UNetModel as UNetModel1f

    torch2of_class_map.update(
        {
            comfy.ldm.modules.diffusionmodules.openaimodel.Upsample: Upsample1f,
            comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel: UNetModel1f,
        }
    )
else:
    print("Dynamic batchsize is not supported in community version.")


register(torch2oflow_class_map=torch2of_class_map)
