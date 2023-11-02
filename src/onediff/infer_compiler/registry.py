"""Registry for compiler inference functions."""
from typing import Callable, Dict, List, Optional, Union
from .utils.args_tree_util import register_args_tree_relaxed_types
from .convert_torch_to_of._globals import load_class_proxies_from_packages
from .convert_torch_to_of.custom_register import (
    register_torch2of_class,
    register_custom_torch2of_func,
)
from pathlib import Path

__all__ = ["register"]


def register(
    package_names: Optional[List[Union[Path, str]]] = None,
    torch2of_class_map: Dict[type, type] = None,
    torch2of_funcs: Optional[List[Callable]] = None,
):
    if package_names:
        load_class_proxies_from_packages(package_names)
        for load_package in package_names:
            if "transformers" in str(load_package):
                register_args_tree_relaxed_types()
                break 

    if torch2of_class_map:
        for torch_cls, of_cls in torch2of_class_map.items():
            register_torch2of_class(torch_cls, of_cls)


    if torch2of_funcs:
        for func in torch2of_funcs:
            register_custom_torch2of_func(func)
