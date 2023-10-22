""" Custom node converters for torch2of.
ONEDIFF_MODEL_CLASS_REPLACEMENT_MAP = { PYTORCH_MODEL_CLASS: ONEFLOW_MODEL_CLASS }
ONEDIFF_CUSTOM_TORCH2OF_FUNC_TYPE_MAP = { Function :  TYPE }
Function: custom function
TYPE: custom function args_0_type : (torch.Tensor, flow.Tensor) -> torch.Tensor
"""

import os
from pathlib import Path
from ..import_tools import (
    print_yellow,
    get_mock_cls_name,
    import_module_from_path,
)
from .register import torch2of
from ._globals import update_class_proxies


def load_custom_node(module_path):
    try:
        module = import_module_from_path(module_path)

        if (
            hasattr(module, "ONEDIFF_MODEL_CLASS_REPLACEMENT_MAP")
            and getattr(module, "ONEDIFF_MODEL_CLASS_REPLACEMENT_MAP") is not None
        ):
            for cls, replacement in module.ONEDIFF_MODEL_CLASS_REPLACEMENT_MAP.items():
                key = get_mock_cls_name(cls)
                update_class_proxies({key: replacement})

        if (
            hasattr(module, "ONEDIFF_CUSTOM_TORCH2OF_FUNC_TYPE_MAP")
            and getattr(module, "ONEDIFF_CUSTOM_TORCH2OF_FUNC_TYPE_MAP") is not None
        ):
            for (
                func,
                args_0_type,
            ) in module.ONEDIFF_CUSTOM_TORCH2OF_FUNC_TYPE_MAP.items():
                if torch2of.registry.get(args_0_type, None) is not None:
                    print_yellow(
                        f"Custom register {func=} {args_0_type=} already exists, skip"
                    )
                    continue
                torch2of.register(args_0_type)(func)
        return module

    except ImportError as e:
        print_yellow(f"ImportError: {e}")

    except Exception as e:
        print_yellow(f"Cannot import {module_path} module for custom register")
        print_yellow(e)


def _init_custom_register():
    custom_nodes_dir = Path(__file__).parents[3] / "onediff_mock_extension"

    custom_node_paths = [
        path
        for path in custom_nodes_dir.iterdir()
        if not path.name.endswith(".py.example")
        and not path.name.endswith(".pyc")
        and not path.name.startswith("__")
    ]
    custom_node_paths.sort()
    for path in custom_node_paths:
        load_custom_node(path)

    custom_register_path = os.getenv("ONEDIFF_CUSTOM_REGISTER_PATH", None)
    if custom_register_path is not None:
        for path in custom_register_path.split(":"):
            load_custom_node(path)


_init_custom_register()
