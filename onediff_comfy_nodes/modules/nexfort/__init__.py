import os

from .hijack_comfyui_instantid import comfyui_instantid_hijacker
from .hijack_ipadapter_plus import ipadapter_plus_hijacker
from .hijack_pulid_comfyui import pulid_comfyui_hijacker
from .hijack_samplers import samplers_hijack

samplers_hijack.hijack(last=False)
ipadapter_plus_hijacker.hijack(last=False)
pulid_comfyui_hijacker.hijack(last=False)
comfyui_instantid_hijacker.hijack(last=False)


# https://github.com/pytorch/pytorch/blob/1edcb31d34ef012d828bb9f39a8aef6020f580b2/aten/src/ATen/cuda/CUDABlas.cpp#L182-L203
if os.getenv("CUBLASLT_WORKSPACE_SIZE") is None:
    os.environ["CUBLASLT_WORKSPACE_SIZE"] = str(1024 * 1024)
