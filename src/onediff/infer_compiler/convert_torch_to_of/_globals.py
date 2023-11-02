"""OneDiff global variables.
- `_ONEDIFF_TORCH_TO_OF_CLASS_MAP`: {torch_module_name: of_module_proxy, ...}
- `_ONEDIFF_LOADED_PACKAGES`: [package_name, ...]
"""
import oneflow as flow
from typing import Dict, List, Union
from pathlib import Path
from ..import_tools import (
    get_classes_in_package,
    print_green,
)

__all__ = ["update_class_proxies", "load_class_proxies_from_packages"]
_ONEDIFF_TORCH_TO_OF_CLASS_MAP = {}
_ONEDIFF_LOADED_PACKAGES = []


def load_class_proxies_from_packages(package_names: List[Union[Path, str]]):
    global _ONEDIFF_TORCH_TO_OF_CLASS_MAP
    global _ONEDIFF_LOADED_PACKAGES

    print_green(f"==> Loading modules: {package_names}")
    of_mds = {}
    # https://docs.oneflow.org/master/cookies/oneflow_torch.html
    with flow.mock_torch.enable(lazy=True):
        for package_name in package_names:
            of_mds.update(get_classes_in_package(package_name))
    print_green(f"Loaded Mock Torch {len(of_mds)} classes: {package_names} <==")
    _ONEDIFF_TORCH_TO_OF_CLASS_MAP.update(of_mds)
    _ONEDIFF_LOADED_PACKAGES.extend(package_names)


def update_class_proxies(class_proxy_dict: Dict[str, type], verbose=True):
    """Update `_ONEDIFF_TORCH_TO_OF_CLASS_MAP` with `class_proxy_dict`.

    example: 
        `class_proxy_dict = {"mock_torch.nn.Conv2d": flow.nn.Conv2d}`

    """
    global _ONEDIFF_TORCH_TO_OF_CLASS_MAP

    _ONEDIFF_TORCH_TO_OF_CLASS_MAP.update(class_proxy_dict)

    if verbose:
        print_green(
            f"==> Loaded Mock Torch {len(class_proxy_dict)} "
            f"classes: {class_proxy_dict.keys()}... "
        )