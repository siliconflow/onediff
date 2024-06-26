import os
import traceback

COMFYUI_ROOT = os.getenv("COMFYUI_ROOT")
from onediff.infer_compiler.backends.oneflow.import_tools import DynamicModuleLoader
from onediff.infer_compiler.backends.oneflow.transform import transform_mgr

from ...sd_hijack_utils import Hijacker

__all__ = ["pulid_comfyui_pt", "pulid_comfyui_of"]

pkg_name = "PuLID_ComfyUI"
pkg_root = os.path.join(COMFYUI_ROOT, "custom_nodes", pkg_name)
is_load_pulid_comfyui_pkg = True
try:
    if os.path.exists(pkg_root):
        pulid_comfyui_pt = DynamicModuleLoader.from_path(pkg_root)
        pulid_comfyui_of = transform_mgr.transform_package(pkg_name)
    else:
        is_load_pulid_comfyui_pkg = False
except Exception as e:
    print(traceback.format_exc())
    print(f"Warning: Failed to load {pkg_root} due to {e}")
    is_load_pulid_comfyui_pkg = False
pulid_comfyui_hijacker = Hijacker()
