"""
github: https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved
commit: 5d875d77fe6e31a4b0bc6dc36f0441eba3b6afe1 
"""
import os
from ..sd_hijack_utils import Hijacker
from onediff.infer_compiler.transform import transform_mgr
from onediff.infer_compiler.import_tools import DynamicModuleLoader
from onediff.infer_compiler.utils.log_utils import logger
COMFYUI_ROOT = os.getenv("COMFYUI_ROOT")
pkg_name = "ComfyUI-AnimateDiff-Evolved"
animatediff_root = os.path.join(COMFYUI_ROOT, "custom_nodes", pkg_name)
load_animatediff_package = True
try:
    animatediff_pt = DynamicModuleLoader.from_path(animatediff_root)
    animatediff_of = transform_mgr.transform_package(pkg_name)
    comfy_of = transform_mgr.transform_package("comfy")
except Exception as e:
   logger.warning(f"Failed to load {pkg_name} from {animatediff_root} due to {e}")
   load_animatediff_package = False

animatediff_hijacker = Hijacker()
