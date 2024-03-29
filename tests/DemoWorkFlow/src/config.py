import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
COMFYUI_ROOT=os.getenv("COMFYUI_ROOT", None)
if COMFYUI_ROOT is None:
    raise RuntimeError("COMFYUI_ROOT is not set in environment variables,please set it first")
custom_nodes = "custom_nodes"
add_path(COMFYUI_ROOT)
add_path(os.path.join(COMFYUI_ROOT, custom_nodes))  

from comfy.cli_args import args
args.gpu_only=True







