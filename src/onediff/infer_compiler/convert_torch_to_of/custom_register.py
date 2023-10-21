""" Custom node converters for torch2of.
ONEDIFF_MODEL_CLASS_REPLACEMENT_MAP = { PYTORCH_MODEL_CLASS: ONEFLOW_MODEL_CLASS }
ONEDIFF_CUSTOM_TORCH2OF_FUNC_TYPE_MAP = { Function :  TYPE }
Function: custom function
TYPE: custom function args_0_type : (torch.Tensor, flow.Tensor) -> torch.Tensor
"""

import os 
import sys
import importlib
import oneflow as flow
from ..import_tools import (
    print_yellow,
    get_mock_cls_name,
)
from .register import torch2of
from ._globals import update_class_proxies,_initial_package_names

def load_custom_node(module_path, ignore=set()):
    module_name = os.path.basename(module_path)
    if os.path.isfile(module_path):
        sp = os.path.splitext(module_path)
        module_name = sp[0]
    try:
        if os.path.isfile(module_path):
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
            module_dir = os.path.split(module_path)[0]
        else:
            module_spec = importlib.util.spec_from_file_location(module_name, os.path.join(module_path, "__init__.py"))
            module_dir = module_path

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)

        if hasattr(module, "ONEDIFF_MODEL_CLASS_REPLACEMENT_MAP") and getattr(module, "ONEDIFF_MODEL_CLASS_REPLACEMENT_MAP") is not None:
            for cls, replacement in module.ONEDIFF_MODEL_CLASS_REPLACEMENT_MAP.items():
                key = get_mock_cls_name(cls)
                update_class_proxies({key: replacement})

        if hasattr(module, "ONEDIFF_CUSTOM_TORCH2OF_FUNC_TYPE_MAP") and getattr(module, "ONEDIFF_CUSTOM_TORCH2OF_FUNC_TYPE_MAP") is not None:
            for func, args_0_type in module.ONEDIFF_CUSTOM_TORCH2OF_FUNC_TYPE_MAP.items():
                # torch2of.registry.
                torch2of.register(args_0_type)(func)
        else:
            print(f"Skip {module_path} module for custom nodes due to the lack of NODE_CLASS_MAPPINGS.")
            return False

    except Exception as e:
        print_yellow(f"Cannot import {module_path} module for custom register")
        print_yellow(e)
        return False
    
module_path = "/home/fengwen/workspace/packages/diffusers/src/mock_extension/mock_comfy" 
load_custom_node(module_path)

module_path = "/home/fengwen/workspace/packages/diffusers/src/mock_extension/mock_diffusers" 
load_custom_node(module_path)
