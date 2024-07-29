from pathlib import Path

# ComfyUI
from comfy import model_management
from folder_paths import get_input_directory

# onediff
from onediff.infer_compiler import oneflow_compile, OneflowCompileOptions
from onediff.infer_compiler.backends.oneflow.transform import torch2oflow
from onediff.optimization.quant_optimizer import quantize_model

# onediff_comfy_nodes
from .model_patcher import state_dict_hook


def compoile_unet(diffusion_model, graph_file):
    offload_device = model_management.unet_offload_device()
    load_device = model_management.get_torch_device()

    print(f" OneDiffCheckpointLoaderSimple load_checkpoint file_path {graph_file}")

    compile_options = OneflowCompileOptions()
    compile_options.graph_file = graph_file
    compile_options.graph_file_device = load_device
    diffusion_model = oneflow_compile(diffusion_model, options=compile_options)

    return diffusion_model


def quantize_unet(diffusion_model, calibrate_info, inplace=True):
    diffusion_model = quantize_model(
        model=diffusion_model, inplace=inplace, calibrate_info=calibrate_info
    )
    return diffusion_model
