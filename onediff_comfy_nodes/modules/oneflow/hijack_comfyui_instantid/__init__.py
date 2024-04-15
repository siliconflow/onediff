from ._config import is_load_comfyui_instantid_pkg, comfyui_instantid_hijacker
if is_load_comfyui_instantid_pkg:
    from .InstantID import *