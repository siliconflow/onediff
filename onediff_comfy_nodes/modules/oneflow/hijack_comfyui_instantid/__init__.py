from ._config import comfyui_instantid_hijacker, is_load_comfyui_instantid_pkg

if is_load_comfyui_instantid_pkg:
    from .InstantID import *
