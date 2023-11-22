import time
import os
import sys
import torch
import oneflow as flow
from typing import Dict, List, Union
from contextlib import contextmanager
from pathlib import Path
from ..import_tools import (
    get_classes_and_package,
    print_green,
)

__all__ = ["transform_mgr", "get_mock_cls_name", "format_package_name"]


def gen_unique_id():
    timestamp = int(time.time() * 1000)
    process_id = os.getpid()
    # TODO(): refine the unique id
    # sequence = str(uuid.uuid4())
    unique_id = f"{timestamp}{process_id}"
    return unique_id


PREFIX = "mock_"
SUFFIX = "_oflow_" + gen_unique_id()


def format_package_name(package_name):
    return PREFIX + package_name + SUFFIX


def get_mock_cls_name(cls) -> str:
    if isinstance(cls, type):
        cls = f"{cls.__module__}.{cls.__name__}"

    pkg_name, cls_ = cls.split(".", 1)

    pkg_name = format_package_name(pkg_name)
    return f"{pkg_name}.{cls_}"


@contextmanager
def onediff_mock_torch():
    # Fixes  check the 'version'  error.
    attr_name = "__version__"
    restore_funcs = []  # Backup
    if hasattr(flow, attr_name) and hasattr(torch, attr_name):
        orig_flow_attr = getattr(flow, attr_name)
        restore_funcs.append(lambda: setattr(flow, attr_name, orig_flow_attr))
        setattr(flow, attr_name, getattr(torch, attr_name))

    # https://docs.oneflow.org/master/cookies/oneflow_torch.html
    with flow.mock_torch.enable(lazy=True):
        yield

    for restore_func in restore_funcs:
        restore_func()


class TransformManager:
    def __init__(self):
        self._torch_to_oflow_cls_map = {}
        self._torch_to_oflow_packages_list = []
        self._packages_map = {}

    def load_class_proxies_from_packages(self, package_names: List[Union[Path, str]]):
        print_green(f"Loading modules: {package_names}")
        of_mds = {}
        with onediff_mock_torch():
            for package_name in package_names:
                classes, pkg = get_classes_and_package(
                    package_name, prefix=PREFIX, suffix=SUFFIX
                )
                of_mds.update(classes)
                self._packages_map[format_package_name(package_name)] = pkg

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

    def transform_cls(self, cls):
        full_cls_name = get_mock_cls_name(cls)
        if full_cls_name in self._torch_to_oflow_cls_map:
            return self._torch_to_oflow_cls_map[full_cls_name]

        raise RuntimeError(
            f"""
            Replace can't find proxy oneflow module for: {cls}. \n 
            You need to register it. 
            """
        )

    def transform_package(self, package_name):
        full_pkg_name = format_package_name(package_name)
        if full_pkg_name in self._packages_map:
            return self._packages_map[full_pkg_name]
        else:
            raise RuntimeError(
                f"""
                Package {package_name} is not registered. \n 
                You need to register it. 
                """
            )

    def transform_package_name(self, package_name):
        full_pkg_name = format_package_name(package_name)
        return full_pkg_name


transform_mgr = TransformManager()
