import os
from ..sd_hijack_utils import Hijacker
from onediff.infer_compiler.transform import transform_mgr
from onediff.infer_compiler.import_tools import DynamicModuleLoader

COMFYUI_ROOT = os.getenv("COMFYUI_ROOT")
pkg_name = "ComfyUI-AnimateDiff-Evolved"
animatediff_root = os.path.join(COMFYUI_ROOT, "custom_nodes", pkg_name)
animatediff_pt = DynamicModuleLoader.from_path(animatediff_root)
animatediff_of = transform_mgr.transform_package("ComfyUI-AnimateDiff-Evolved")
animatediff_hijacker = Hijacker()
