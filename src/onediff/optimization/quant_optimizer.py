import time
import torch
import torch.nn as nn
from copy import deepcopy
from ..infer_compiler.utils.log_utils import logger
from ..infer_compiler.utils.version_util import is_quantization_enabled
from ..infer_compiler.utils.cost_util import cost_cnt
from ..infer_compiler.utils.module_operations import modify_sub_module
from ..infer_compiler.transform.manager import transform_mgr


__all__ = ["quantize_model", "varify_can_use_quantization"]


def varify_can_use_quantization():
    if not is_quantization_enabled():
        logger.warning(f"OneDiff Quantization can't be used.")
        return False
    return True


@cost_cnt(debug=transform_mgr.debug_mode)
def quantize_model(
    model,  # diffusion_model
    quantize_conv=True,
    quantize_linear=True,
    bits=8,
    *,
    inplace=True,
    calibrate_info: dict = None,
):
    """Quantize a model. inplace=True will modify the model in-place."""
    start_time = time.time()
    if varify_can_use_quantization() is False:
        return model

    from onediff_quant.utils import symm_quantize_sub_module, find_quantizable_modules
    from onediff_quant.utils import get_quantize_module
    from onediff_quant import Quantizer

    quantize_conv_cnt, quantize_linear_cnt = 0, 0

    if not inplace:
        model = deepcopy(model)

    def no_quantizable(sub_module_name):
        if calibrate_info is not None:
            conf = calibrate_info.get(sub_module_name, None)
            if conf is None:
                return True
            return False
        else:
            return False

    def apply_quantization_to_modules(quantizable_modules):
        nonlocal model, quantize_conv_cnt, quantize_linear_cnt

        for sub_module_name, sub_mod in quantizable_modules.items():
            if no_quantizable(sub_module_name):
                continue

            if isinstance(sub_mod, nn.Conv2d):
                quantize_conv_cnt += 1
            elif isinstance(sub_mod, nn.Linear):
                quantize_linear_cnt += 1

            quantizer = Quantizer()
            quantizer.configure(bits=bits, perchannel=True)
            quantizer.find_params(sub_mod.weight.float(), weight=True)
            shape = [-1] + [1] * (len(sub_mod.weight.shape) - 1)
            scale = quantizer.scale.reshape(*shape)
            symm_quantize_sub_module(
                model, sub_module_name, scale, quantizer.maxq, save_as_float=False
            )

            input_scale = 0
            input_zero_point = 0
            weight_scale = scale.reshape(-1).tolist()
            sub_calibrate_info = [input_scale, input_zero_point, weight_scale]

            sub_mod = get_quantize_module(
                sub_mod,
                sub_module_name,
                sub_calibrate_info,
                fake_quant=False,
                static=False,
                nbits=bits,
            )

            modify_sub_module(model, sub_module_name, sub_mod)

    if quantize_conv:
        conv_modules = find_quantizable_modules(model, module_cls=[nn.Conv2d])
        apply_quantization_to_modules(conv_modules)
        logger.debug(f"{len(conv_modules)=}")

    if quantize_linear:
        linear_modules = find_quantizable_modules(model, module_cls=[nn.Linear])
        apply_quantization_to_modules(linear_modules)
        logger.debug(f"{len(linear_modules)=}")

    logger.info(
        f"Quantized model {type(model)} successfully! \n"
        + f"Quantized conv: {quantize_conv_cnt} \n"
        + f"Quantized linear: {quantize_linear_cnt} \n"
        + f"Time: {time.time() - start_time:.4f}s \n"
    )

    return model

