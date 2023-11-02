import os
import oneflow as flow
from typing import Dict, List, Union
from pathlib import Path
from ..import_tools import (
    get_classes_in_package,
    print_green,
)

__all__ = [
    "update_class_proxies", "load_class_proxies_from_packages"
]
# Dictionary containing class proxies from various packages
_ONEDIFF_CLASS_PROXIES_FROM_VARIOUS_PACKAGES = {}
_ONEDIFF_LOADED_PACKAGES = []

def load_class_proxies_from_packages(package_names: List[Union[Path,str]]):
    global _ONEDIFF_CLASS_PROXIES_FROM_VARIOUS_PACKAGES

    print_green(f"==> Loading modules: {package_names}")
    # https://docs.oneflow.org/master/cookies/oneflow_torch.html
    __of_mds = {}
    with flow.mock_torch.enable(lazy=True):
        for package_name in package_names:
            __of_mds.update(get_classes_in_package(package_name))

    print_green(f" 🚀 Loaded Mock Torch {len(__of_mds)} classes: {package_names} 🚀 <== ")

    _ONEDIFF_CLASS_PROXIES_FROM_VARIOUS_PACKAGES.update(__of_mds)
    _ONEDIFF_LOADED_PACKAGES.extend(package_names)


def update_class_proxies(class_proxy_dict: Dict[str, type]):
    """
    Update the CLASS_PROXIES_FROM_VARIOUS_PACKAGES with the given class_proxy_dict.

    Args:
        class_proxy_dict (Dict[str, type]): a dictionary of class proxies obtained from various packages.
        
        example: class_proxy_dict = {
             "mock_torch.nn.linear": flow.nn.Linear,
        }
    """
    global _ONEDIFF_CLASS_PROXIES_FROM_VARIOUS_PACKAGES
    for module_name, module_proxy in class_proxy_dict.items():
        _ONEDIFF_CLASS_PROXIES_FROM_VARIOUS_PACKAGES[module_name] = module_proxy
    print_green(
        f" Loaded expand {len(class_proxy_dict)} classes: {class_proxy_dict.keys()} <== "
    )



