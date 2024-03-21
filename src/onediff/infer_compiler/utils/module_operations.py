import torch
import torch.nn as nn

__all__ = ["get_sub_module", "modify_sub_module", "quantize_sub_module"]


def get_sub_module(module, sub_module_name) -> nn.Module:
    """Get a submodule of a module using dot-separated names.

    Args:
        module (nn.Module): The base module.
        sub_module_name (str): Dot-separated name of the submodule.

    Returns:
        nn.Module: The requested submodule.
    """
    if sub_module_name == "":
        return module
    
    parts = sub_module_name.split(".")
    current_module = module

    for part in parts:
        try:
            if part.isdigit():
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
        except (IndexError, AttributeError):
            raise ModuleNotFoundError(f"Submodule {part} not found.")

    return current_module


def modify_sub_module(module, sub_module_name, new_value):
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


def quantize_sub_module(
    model, sub_name, sub_module, bits=8, maxq=127, fake_quant=True, save_as_float=False
):
    from onediff_quant.quantization import Quantizer
    from onediff_quant.utils import (
        symm_quantize,
        fake_symm_quantize,
        get_quantize_module,
    )

    if sub_module is None:
        sub_module = get_sub_module(model, sub_name)

    quantizer = Quantizer()
    quantizer.configure(bits=bits, perchannel=True)
    quantizer.find_params(sub_module.weight.float(), weight=True)
    shape = [-1] + [1] * (len(sub_module.weight.shape) - 1)
    scale = quantizer.scale.reshape(*shape)

    org_weight_data = sub_module.weight.data
    org_requires_grad = sub_module.weight.requires_grad

    # save_as_float = False
    sub_module.weight.requires_grad = False
    input_scale_and_zero_point = [None, None]

    if fake_quant or save_as_float:
        sub_module.weight.data = fake_symm_quantize(
            sub_module.weight.data, scale.to(sub_module.weight.data.device), maxq,
        )
    else:
        sub_module.weight.data = symm_quantize(
            sub_module.weight.data, scale.to(sub_module.weight.data.device), maxq,
        )
    quant_module = get_quantize_module(
        sub_module,
        sub_name,
        input_scale_and_zero_point + [scale.reshape(-1).tolist()],
        fake_quant,  # fake_quant
        False,
        bits,
    )
    modify_sub_module(model, sub_name, quant_module)

    def restore():
        sub_module.weight.data = org_weight_data
        sub_module.weight.requires_grad = org_requires_grad
        modify_sub_module(model, sub_name, sub_module)

    return restore