import folder_paths
from pathlib import Path
from comfy import model_management
from .model_patcher import OneFlowSpeedUpModelPatcher
from .graph_path import generate_graph_path
from .loader_sample_tools import compoile_unet, quantize_unet
from .._config import _USE_UNET_INT8, ONEDIFF_QUANTIZED_OPTIMIZED_MODELS

import torch


def onediff_load_quant_checkpoint_advanced(
    ckpt_name, model_path, need_compile, vae_speedup, modelpatcher, vae
):
    """
    Loads a quantized and potentially optimized model checkpoint, applies quantization,
    optionally compiles the model, and applies VAE speedup if enabled.
    """
    ckpt_name = f"{ckpt_name}_quant_{model_path}"
    model_path = (
        Path(folder_paths.models_dir) / ONEDIFF_QUANTIZED_OPTIMIZED_MODELS / model_path
    )
    graph_file = generate_graph_path(ckpt_name, modelpatcher.model)

    calibrate_info = torch.load(model_path)

    load_device = model_management.get_torch_device()
    diffusion_model = modelpatcher.model.diffusion_model.to(load_device)
    quant_unet = quantize_unet(
        diffusion_model=diffusion_model, inplace=True, calibrate_info=calibrate_info,
    )
    modelpatcher.model.diffusion_model = quant_unet

    if need_compile:
        # compiled_unet = compoile_unet(
        #     modelpatcher.model.diffusion_model, graph_file
        # )
        # modelpatcher.model.diffusion_model = compiled_unet
        offload_device = model_management.unet_offload_device()
        modelpatcher = OneFlowSpeedUpModelPatcher(
            modelpatcher.model,
            load_device=model_management.get_torch_device(),
            offload_device=offload_device,
            use_graph=True,
            graph_path=graph_file,
            graph_device=model_management.get_torch_device(),
        )

    if vae_speedup == "enable":
        file_path = generate_graph_path(ckpt_name, vae.first_stage_model)
        vae.first_stage_model = oneflow_compile(
            vae.first_stage_model,
            use_graph=True,
            options={
                "graph_file": file_path,
                "graph_file_device": model_management.get_torch_device(),
            },
        )

    # set inplace update
    modelpatcher.weight_inplace_update = True
    return modelpatcher, vae
