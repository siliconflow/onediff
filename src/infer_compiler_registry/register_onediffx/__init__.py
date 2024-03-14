from onediff.infer_compiler.transform import register

from onediffx.deepcache.models.unet_2d_blocks import CrossAttnUpBlock2D, UpBlock2D

from .unet_2d_blocks import CrossAttnUpBlock2D as CrossAttnUpBlock2DOflow
from .unet_2d_blocks import UpBlock2D as UpBlock2DOflow

torch2oflow_class_map = {
  CrossAttnUpBlock2D: CrossAttnUpBlock2DOflow,
  UpBlock2D: UpBlock2DOflow,
}

register(torch2oflow_class_map=torch2oflow_class_map)
