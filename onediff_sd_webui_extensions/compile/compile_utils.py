import os
from typing import Dict

# import modules.shared as shared
import warnings
from typing import Union, Dict
from pathlib import Path

from .compile_ldm import compile_ldm_unet
from .compile_sgm import compile_sgm_unet
from .onediff_compiled_graph import OneDiffCompiledGraph
from ldm.modules.diffusionmodules.openaimodel import UNetModel as UNetModelLDM
from sgm.modules.diffusionmodules.openaimodel import UNetModel as UNetModelSGM
from onediff.optimization.quant_optimizer import (
    quantize_model,
    varify_can_use_quantization,
)
from onediff.utils import logger
from onediff_shared import graph_dict

from modules.sd_models import select_checkpoint


def compile_unet(
    unet_model, quantization=False, *, options=None,
):
    if isinstance(unet_model, UNetModelLDM):
        compiled_unet = compile_ldm_unet(unet_model, options=options)
    elif isinstance(unet_model, UNetModelSGM):
        compiled_unet = compile_sgm_unet(unet_model, options=options)
    else:
        warnings.warn(
            f"Unsupported model type: {type(unet_model)} for compilation , skip",
            RuntimeWarning,
        )
        compiled_unet = unet_model
    # In OneDiff Community, quantization can be True when called by api
    if quantization and varify_can_use_quantization():
        calibrate_info = get_calibrate_info(
            f"{Path(select_checkpoint().filename).stem}_sd_calibrate_info.txt"
        )
        compiled_unet = quantize_model(
            compiled_unet, inplace=False, calibrate_info=calibrate_info
        )
    return compiled_unet


def get_calibrate_info(filename: str) -> Union[None, Dict]:
    calibration_path = Path(select_checkpoint().filename).parent / filename
    if not calibration_path.exists():
        return None

    logger.info(f"Got calibrate info at {str(calibration_path)}")
    calibrate_info = {}
    with open(calibration_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            items = line.split(" ")
            calibrate_info[items[0]] = [
                float(items[1]),
                int(items[2]),
                [float(x) for x in items[3].split(",")],
            ]
    return calibrate_info


def get_compiled_graph(sd_model, quantization) -> OneDiffCompiledGraph:
    if sd_model.sd_model_hash in graph_dict:
        return graph_dict[sd_model.sd_model_hash]
    else:
        compiled_unet = compile_unet(
            sd_model.model.diffusion_model, quantization=quantization
        )
        return OneDiffCompiledGraph(sd_model, compiled_unet, quantization)
