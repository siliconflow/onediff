"""
github: https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved
commit: 4b31c0819d361e58f222cee3827079b3a6b6f966 
"""
import os
from ..sd_hijack_utils import Hijacker
from onediff.infer_compiler.transform import transform_mgr
from onediff.infer_compiler.import_tools import DynamicModuleLoader

COMFYUI_ROOT = os.getenv("COMFYUI_ROOT")
pkg_name = "ComfyUI-AnimateDiff-Evolved"
animatediff_root = os.path.join(COMFYUI_ROOT, "custom_nodes", pkg_name)
animatediff_pt = DynamicModuleLoader.from_path(animatediff_root)
animatediff_of = transform_mgr.transform_package("ComfyUI-AnimateDiff-Evolved")
comfy_of = transform_mgr.transform_package("comfy")
animatediff_hijacker = Hijacker()
