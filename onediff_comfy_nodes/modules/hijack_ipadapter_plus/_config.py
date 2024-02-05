import os 
COMFYUI_ROOT = os.getenv("COMFYUI_ROOT")
from onediff.infer_compiler.transform import transform_mgr
from onediff.infer_compiler.import_tools import DynamicModuleLoader
from ..sd_hijack_utils import Hijacker
__all__ = ["ipadapter_plus_pt", "ipadapter_plus_of"]

pkg_name = "ComfyUI_IPAdapter_plus"
pkg_root = os.path.join(COMFYUI_ROOT, "custom_nodes", pkg_name)
ipadapter_plus_pt = DynamicModuleLoader.from_path(pkg_root)
ipadapter_plus_of = transform_mgr.transform_package(pkg_name)
ipadapter_plus_hijacker = Hijacker()