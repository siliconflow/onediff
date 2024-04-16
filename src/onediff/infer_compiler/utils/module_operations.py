import torch
import torch.nn as nn

__all__ = ["get_sub_module", "modify_sub_module"]


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
