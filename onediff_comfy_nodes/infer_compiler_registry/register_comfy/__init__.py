from onediff.infer_compiler import register
from nodes import *  # must imported before import comfy
from pathlib import Path

comfy_path = Path(os.path.abspath(__file__)).parents[4] / "comfy"
register(package_names=[comfy_path])
import comfy
from .attention import CrossAttention as CrossAttention1f
from .attention import SpatialTransformer as SpatialTransformer1f
from .linear import Linear as Linear1f
from .openaimodel import Upsample as Upsample1f
from .openaimodel import UNetModel as UNetModel1f

torch2of_class_map = {
    comfy.ldm.modules.attention.CrossAttention: CrossAttention1f,
    comfy.ldm.modules.attention.SpatialTransformer: SpatialTransformer1f,
    comfy.ops.Linear: Linear1f,
    comfy.ldm.modules.diffusionmodules.openaimodel.Upsample: Upsample1f,
    comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel: UNetModel1f,
}


register(torch2oflow_class_map=torch2of_class_map)
