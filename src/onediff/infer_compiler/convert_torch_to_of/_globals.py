import os
from typing import Dict, List
import oneflow as flow
from ..import_tools import (
    get_classes_in_package,
    print_green,
    print_red,
)

__all__ = [
    "update_class_proxies",
]


def __load_class_proxies(package_names: List[str]):
    print_green(f"==> Loading modules: {package_names}")
    # https://docs.oneflow.org/master/cookies/oneflow_torch.html
    __of_mds = {}
    with flow.mock_torch.enable(lazy=True):
        for package_name in package_names:
            __of_mds.update(get_classes_in_package(package_name))

    print_green(f" 🚀 Loaded Mock Torch {len(__of_mds)} classes: {package_names} 🚀 <== ")
    return __of_mds
_initial_package_names = os.getenv(
    "ONEDIFF_INITIAL_PACKAGE_NAMES_FOR_CLASS_PROXIES", "diffusers,transformers"
).split(",")
# Dictionary containing class proxies from various packages
_ONEDIFF_CLASS_PROXIES_FROM_VARIOUS_PACKAGES = __load_class_proxies(
    _initial_package_names
)  # export ONEDIFF_INITIAL_PACKAGE_NAMES_FOR_CLASS_PROXIES="diffusers,comfyui"
_WARNING_MSG = set()


def update_class_proxies(class_proxy_dict: Dict[str, type]):
    """
    Update the CLASS_PROXIES_FROM_VARIOUS_PACKAGES with the given class_proxy_dict.

    Args:
        class_proxy_dict (Dict[str, type]): a dictionary of class proxies obtained from various packages.
        
        example: class_proxy_dict = {
             "mock_torch.nn.linear": flow.nn.Linear,
        }
    """
    for module_name, module_proxy in class_proxy_dict.items():
        _ONEDIFF_CLASS_PROXIES_FROM_VARIOUS_PACKAGES[module_name] = module_proxy
    print_green(
        f" 🚀 Loaded Mock Torch {len(class_proxy_dict)} classes: {class_proxy_dict.keys()} 🚀 <== "
    )
