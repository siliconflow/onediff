import os
import traceback

COMFYUI_ROOT = os.getenv("COMFYUI_ROOT")
from onediff.utils.import_utils import DynamicModuleLoader

from ...sd_hijack_utils import Hijacker

__all__ = ["comfyui_instantid"]

pkg_name = "ComfyUI_InstantID"
pkg_root = os.path.join(COMFYUI_ROOT, "custom_nodes", pkg_name)
is_load_comfyui_instantid_pkg = True
try:
    if os.path.exists(pkg_root):
        comfyui_instantid = DynamicModuleLoader.from_path(pkg_root)
    else:
        is_load_comfyui_instantid_pkg = False
except Exception as e:
    print(traceback.format_exc())
    print(f"Warning: Failed to load {pkg_root} due to {e}")
    is_load_comfyui_instantid_pkg = False
comfyui_instantid_hijacker = Hijacker()
