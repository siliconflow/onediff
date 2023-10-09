import os
import oneflow as flow
from onediff.infer_compiler.import_tools import get_classes_in_package, print_green, print_red 
from typing import Dict

__all__ = ["PROXY_OF_MDS", "WARNING_MSG", "add_to_proxy_of_mds"]


# {name: md} proxy of oneflow modules
def __init_of_mds(package_names: list[str]):
    print_red(f'==> Loading modules: {package_names}')
    # https://docs.oneflow.org/master/cookies/oneflow_torch.html
    __of_mds = {}
    with flow.mock_torch.enable(lazy=True):
        for package_name in package_names:
            __of_mds.update(get_classes_in_package(package_name))

    print_green(f' 🚀 Loaded Mock Torch {len(__of_mds)} classes: {package_names} 🚀 <== ')
    return __of_mds


package_names = os.getenv("INIT_OF_MDS", "diffusers")
PROXY_OF_MDS = __init_of_mds(
    package_names.split(",")
)  # export INIT_OF_MDS="diffusers,comfyui"
WARNING_MSG = set()


def add_to_proxy_of_mds(new_module_proxies: Dict[str, type]):
    """Add new module proxies to PROXY_OF_MDS"""
    for module_name, module_proxy in new_module_proxies.items():
        PROXY_OF_MDS[module_name] = module_proxy
    print_green(
        f"Added {len(new_module_proxies)} module proxies: {', '.join(new_module_proxies.keys())}"
    )
