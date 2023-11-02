import os
from pathlib import Path
import inspect
from ..import_tools import (
    print_yellow,
    print_green,
    get_mock_cls_name
)
from .register import torch2onef
from ._globals import update_class_proxies

__all__ = ["register_custom_torch2of_class", "register_custom_torch2of_func"]
# ONEDIFF_TORCH_TO_ONEF_CLASS_MAP
def register_torch2of_class(cls:type, replacement:type):
    try:
        key = get_mock_cls_name(cls)
        update_class_proxies({key: replacement})
    except Exception as e:
        print_yellow(f"Cannot register {cls=} {replacement=}. {e=}")


def register_custom_torch2of_func(func, first_param_type=None):
    if first_param_type is None:
        params = inspect.signature(func).parameters
        first_param_type = params[list(params.keys())[0]].annotation
        if first_param_type == inspect._empty:
            print_yellow(f"Cannot register {func=} {first_param_type=}.")
            return 
    try:
        torch2onef.register(first_param_type)(func)
        print_green(f"Register {func=} {first_param_type=}")
    except Exception as e:
        print_yellow(f"Cannot register {func=} {first_param_type=}. {e=}")


# # def ONEDIFF_CUSTOM_TORCH2OF_FUNC_TYPE_MAP(func, args_0_type):


# def load_custom_type_map(module_path):
#     try:
#         module = import_module_from_path(module_path)

#         if (
#             hasattr(module, "ONEDIFF_TORCH_TO_ONEF_CLASS_MAP")
#             and getattr(module, "ONEDIFF_TORCH_TO_ONEF_CLASS_MAP") is not None
#         ):
#             for cls, replacement in module.ONEDIFF_TORCH_TO_ONEF_CLASS_MAP.items():
#                 key = get_mock_cls_name(cls)
#                 update_class_proxies({key: replacement})

#         if (
#             hasattr(module, "ONEDIFF_CUSTOM_TORCH2OF_FUNC_TYPE_MAP")
#             and getattr(module, "ONEDIFF_CUSTOM_TORCH2OF_FUNC_TYPE_MAP") is not None
#         ):
#             for (
#                 func,
#                 args_0_type,
#             ) in module.ONEDIFF_CUSTOM_TORCH2OF_FUNC_TYPE_MAP.items():
#                 if torch2onef.registry.get(args_0_type, None) is not None:
#                     print_yellow(
#                         f"Custom register {func=} {args_0_type=} already exists, skip"
#                     )
#                     continue
#                 torch2onef.register(args_0_type)(func)
#         return module

#     except ImportError as e:
#         print_yellow(f"ImportError: {e}")

#     except Exception as e:
#         print_yellow(f"Cannot import {module_path} module for custom register")
#         print_yellow(e)


# def _init_custom_register():
#     custom_nodes_dir = Path(__file__).parents[3] / "onediff_mock_extension"

#     custom_node_paths = [
#         path
#         for path in custom_nodes_dir.iterdir()
#         if not path.name.endswith(".py.example")
#         and not path.name.endswith(".pyc")
#         and not path.name.startswith("__")
#     ]
#     custom_node_paths.sort()
#     for path in custom_node_paths:
#         load_custom_type_map(path)

#     custom_register_path = os.getenv("ONEDIFF_CUSTOM_REGISTER_PATH", None)
#     if custom_register_path is not None:
#         for path in custom_register_path.split(":"):
#             load_custom_type_map(path)


# _init_custom_register()
