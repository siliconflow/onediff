"""A module for registering custom torch2oflow functions and classes."""
import warnings
import inspect
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
from ..import_tools import (
    get_mock_entity_name,
    import_module_from_path,
)
from .manager import transform_mgr
from .builtin_transform import torch2oflow

__all__ = ["register"]


def register_torch2oflow_class(cls: type, replacement: type, verbose=True):
    try:
        key = get_mock_entity_name(cls)
        transform_mgr.update_class_proxies({key: replacement}, verbose=verbose)

    except Exception as e:
        print_yellow(f"Cannot register {cls=} {replacement=}. {e=}")


def register_torch2oflow_func(func, first_param_type=None, verbose=True):
    if first_param_type is None:
        params = inspect.signature(func).parameters
        first_param_type = params[list(params.keys())[0]].annotation
        if first_param_type == inspect._empty:
            print_yellow(f"Cannot register {func=} {first_param_type=}.")
    try:
        torch2oflow.register(first_param_type)(func)
        if verbose:
            print_green(f"Register {func=} {first_param_type=}")
    except Exception as e:
        print_yellow(f"Cannot register {func=} {first_param_type=}. {e=}")


def set_default_registry():
    if len(transform_mgr._torch_to_oflow_packages_list) > 0:
        return  # already set

    # compiler_registry_path
    registry_path = Path(__file__).parents[3] / "infer_compiler_registry"

    try:
        import_module_from_path(registry_path / "register_diffusers")
    except Exception as e:
        warnings.warn(f"Failed to register_diffusers {e=}")

    try:
        import_module_from_path(registry_path / "register_diffusers_quant")
    except Exception as e:
        warnings.warn(f"Failed to register_diffusers_quant {e=}")


def register(
    *,
    package_names: Optional[List[Union[Path, str]]] = None,
    torch2oflow_class_map: Optional[Dict[type, type]] = None,
    torch2oflow_funcs: Optional[List[Callable]] = None,
):
    if package_names:
        transform_mgr.load_class_proxies_from_packages(package_names)

    if torch2oflow_class_map:
        for torch_cls, of_cls in torch2oflow_class_map.items():
            register_torch2oflow_class(torch_cls, of_cls)

    if torch2oflow_funcs:
        for func in torch2oflow_funcs:
            register_torch2oflow_func(func)
