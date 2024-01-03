import time
import torch.nn as nn
from copy import  deepcopy
from ..infer_compiler.utils.log_utils import logger
from ..infer_compiler.utils.version_util import (
    get_support_message,
    is_quantization_enabled,
)
from ..infer_compiler.utils.cost_util import cost_cnt
from ..infer_compiler.transform.manager import transform_mgr


__all__ = ["quantize_model", "varify_can_use_quantization"]


def varify_can_use_quantization():
    if not is_quantization_enabled():
        message = get_support_message()
        logger.warn(message)
        return False
    return True


def _modify_sub_module(module, sub_module_name, new_value, inplace=True):
    """Modify a submodule of a module using dot-separated names.

    Args:
        module (nn.Module): The base module.
        sub_module_name (str): Dot-separated name of the submodule.
        new_value: The new value to assign to the submodule.

    """
    parts = sub_module_name.split(".")
    current_module = module
    for i, part in enumerate(parts):
        try:
            if part.isdigit():
                if i == len(parts) - 1:
                    current_module[int(part)] = new_value
                else:
                    current_module = current_module[int(part)]
            else:
                if i == len(parts) - 1:
                    setattr(current_module, part, new_value)
                else:
                    current_module = getattr(current_module, part)
        except (IndexError, AttributeError):
            raise ModuleNotFoundError(f"Submodule {part} not found.")


@cost_cnt(debug=transform_mgr.debug_mode)
def quantize_model(
    model,  # diffusion_model
    quantize_conv=True,
    quantize_linear=True,
    bits=8,
    *,
    inplace=True,
):
    """Quantize a model. inplace=True will modify the model in-place."""
    start_time = time.time()
    if varify_can_use_quantization() is False:
        return model
    
    from torch._dynamo import allow_in_graph as maybe_allow_in_graph
    from diffusers_quant.utils import symm_quantize_sub_module, find_quantizable_modules
    from diffusers_quant.utils import get_quantize_module
    from diffusers_quant import Quantizer

    if not inplace:
        model = deepcopy(model)

    
    def apply_quantization_to_modules(quantizable_modules):
        nonlocal model
        for sub_module_name, sub_mod in quantizable_modules.items():

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
                convert_fn=maybe_allow_in_graph,
            )
            _modify_sub_module(model, sub_module_name, sub_mod)



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
        + f"Time: {time.time() - start_time:.4f}s"
    )

    return model
