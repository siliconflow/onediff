"""Registry for compiler inference functions."""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
from .import_tools import import_module_from_path
from .utils.args_tree_util import register_args_tree_relaxed_types
from .convert_torch_to_of._globals import (
    load_class_proxies_from_packages,
    _ONEDIFF_LOADED_PACKAGES,
)
from .convert_torch_to_of.custom_register import (
    register_torch2of_class,
    register_custom_torch2of_func,
)

__all__ = ["register"]


def set_default_config():
    global _ONEDIFF_LOADED_PACKAGES
    if _ONEDIFF_LOADED_PACKAGES:
        return  # already set

    # compiler_registry_path
    registry_path = Path(__file__).parents[2] / "compiler_registry"

    try:
        import_module_from_path(registry_path / "register_diffusers")
    except Exception as e:
        print(f"Failed to import register_diffusers_quant {e=}")
    try:
        import_module_from_path(registry_path / "register_diffusers_quant")
    except:
        print(f"Failed to import register_diffusers_quant {e=}")


def register(
    *,
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
