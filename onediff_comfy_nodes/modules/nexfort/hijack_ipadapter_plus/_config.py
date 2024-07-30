import os
import traceback

COMFYUI_ROOT = os.getenv("COMFYUI_ROOT")
from onediff.utils.import_utils import DynamicModuleLoader

from ...sd_hijack_utils import Hijacker

__all__ = ["ipadapter_plus", "ipadapter_plus_hijacker"]

pkg_name = "ComfyUI_IPAdapter_plus"
pkg_root = os.path.join(COMFYUI_ROOT, "custom_nodes", pkg_name)
is_load_ipadapter_plus_pkg = True
try:
    if os.path.exists(pkg_root):
        ipadapter_plus = DynamicModuleLoader.from_path(pkg_root)
    else:
        is_load_ipadapter_plus_pkg = False
except Exception as e:
    print(traceback.format_exc())
    print(f"Warning: Failed to load {pkg_root} due to {e}")
    is_load_ipadapter_plus_pkg = False
ipadapter_plus_hijacker = Hijacker()
