import oneflow as flow
from typing import Dict, List, Union
from pathlib import Path
from ..import_tools import (
    get_classes_in_package,
    print_green,
)

__all__ = ["transform_mgr"]

class TransformManager():
    def __init__(self):
        self._torch_to_oflow_cls_map = {}
        self._torch_to_oflow_packages_list = []

    def load_class_proxies_from_packages(self, package_names: List[Union[Path, str]]):
        print_green(f"Loading modules: {package_names}")
        of_mds = {}
        # https://docs.oneflow.org/master/cookies/oneflow_torch.html
        with flow.mock_torch.enable(lazy=True):
            for package_name in package_names:
                of_mds.update(get_classes_in_package(package_name))
        print_green(f"Loaded Mock Torch {len(of_mds)} classes: {package_names}")
        self._torch_to_oflow_cls_map.update(of_mds)
        self._torch_to_oflow_packages_list.extend(package_names)
    
    
    def update_class_proxies(self, class_proxy_dict: Dict[str, type], verbose=True):
        """Update `_torch_to_oflow_cls_map` with `class_proxy_dict`.
    
        example: 
            `class_proxy_dict = {"mock_torch.nn.Conv2d": flow.nn.Conv2d}`
    
        """
        self._torch_to_oflow_cls_map.update(class_proxy_dict)
    
        if verbose:
            print_green(
                f"Loaded Mock Torch {len(class_proxy_dict)} "
                f"classes: {class_proxy_dict.keys()}... "
            )

    def transform_cls(self, full_cls_name):
        if full_cls_name in self._torch_to_oflow_cls_map:
            return self._torch_to_oflow_cls_map[full_cls_name]

        raise RuntimeError(
            f"""
            Replace can't find proxy oneflow module for: {str(cls)}. \n 
            You need to register it. 
            """
        )

transform_mgr = TransformManager()
