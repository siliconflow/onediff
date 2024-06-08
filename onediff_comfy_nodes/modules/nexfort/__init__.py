from .hijack_samplers import samplers_hijack
from .hijack_ipadapter_plus import ipadapter_plus_hijacker
from .hijack_pulid_comfyui import pulid_comfyui_hijacker
from .hijack_comfyui_instantid import comfyui_instantid_hijacker

samplers_hijack.hijack(last=False)
ipadapter_plus_hijacker.hijack(last=False)
pulid_comfyui_hijacker.hijack(last=False)
comfyui_instantid_hijacker.hijack(last=False)
