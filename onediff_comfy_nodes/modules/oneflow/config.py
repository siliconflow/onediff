"""Config file for comfyui-speedup, which will automatically"""
import os
import sys
from pathlib import Path

from onediff.infer_compiler.backends.oneflow.utils.version_util import (
    is_community_version,
)

# Delete the diffusers module
if "diffusers" in sys.modules:
    del sys.modules["diffusers"]

# Set up paths
ONEDIFF_QUANTIZED_OPTIMIZED_MODELS = "onediff_quant"
COMFYUI_ROOT = os.getenv("COMFYUI_ROOT")

custom_nodes_path = os.path.join(COMFYUI_ROOT, "custom_nodes")
infer_compiler_registry_path = os.path.join(
    os.path.dirname(__file__), "infer_compiler_registry"
)

# Add paths to sys.path if not already there
if custom_nodes_path not in sys.path:
    sys.path.append(custom_nodes_path)

if infer_compiler_registry_path not in sys.path:
    sys.path.append(infer_compiler_registry_path)

# infer_compiler_registry/register_comfy
import register_comfy  # load plugins

_USE_UNET_INT8 = not is_community_version()
if _USE_UNET_INT8:
    # infer_compiler_registry/register_onediff_quant
    import register_onediff_quant  # load plugins
    from folder_paths import folder_names_and_paths, models_dir, supported_pt_extensions

    unet_int8_model_dir = Path(models_dir) / "unet_int8"
    unet_int8_model_dir.mkdir(parents=True, exist_ok=True)
    folder_names_and_paths["unet_int8"] = (
        [str(unet_int8_model_dir)],
        supported_pt_extensions,
    )

    opt_models_dir = Path(models_dir) / ONEDIFF_QUANTIZED_OPTIMIZED_MODELS
    opt_models_dir.mkdir(parents=True, exist_ok=True)

    folder_names_and_paths[ONEDIFF_QUANTIZED_OPTIMIZED_MODELS] = (
        [str(opt_models_dir)],
        supported_pt_extensions,
    )
