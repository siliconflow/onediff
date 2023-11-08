import os
import sys
import inspect
import types
import pkgutil
import importlib
from typing import Dict, Union
from types import ModuleType
from pathlib import Path
from .copier import PackageCopier
from .copy_utils import get_proxy_func_name


PREFIX = "mock_"
SUFFIX = "_oneflow"

__all__ = ["get_classes_in_package", "get_mock_cls_name", "import_module_from_path"]


def import_module_from_path(module_path: Union[str, Path]) -> ModuleType:
    if isinstance(module_path, Path):
        module_path = str(module_path)
    module_name = os.path.basename(module_path)
    if os.path.isfile(module_path):
        sp = os.path.splitext(module_path)
        module_name = sp[0]

    if os.path.isfile(module_path):
        module_spec = importlib.util.spec_from_file_location(module_name, module_path)
        module_dir = os.path.split(module_path)[0]
    else:
        module_spec = importlib.util.spec_from_file_location(
            module_name, os.path.join(module_path, "__init__.py")
        )
        module_dir = module_path

    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module

def import_submodules(package, recursive=True):
    """Import all submodules of a module, recursively, including subpackages"""
    if isinstance(package, str):
        package = importlib.import_module(package)

    for _, full_name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        try:
            # TODO(): Avoid copy, rename comfy.x.x.x with mocked_comfy.x.x.x
            good_import = importlib.import_module(full_name)
            yield good_import

        except Exception as e:
            # logger.debug(f"Failed to import {full_name}: {e}")
            pass  # ignore


        if recursive and is_pkg:
            try:
                yield from import_submodules(full_name)
            except Exception as e:
                pass  # ignore


def get_classes_in_package(package, base_class=None) -> Dict[str, type]:
    """
    Get all classes in a package and its submodules.

    Args:
        package (str or module): The package to search for classes.
        base_class (type, optional): The base class to filter classes by.

    Returns:
        dict: A dictionary mapping full class names to class objects.
    """
    if isinstance(package, (str, Path)):
        copier = PackageCopier(package, prefix=PREFIX, suffix=SUFFIX)
        copier()  # copy package
        package = copier.get_import_module()

    class_dict = {}

    for module in import_submodules(package):
        try:
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if inspect.isclass(obj) and (
                    base_class is None or issubclass(obj, base_class)
                ):
                    full_name = f"{obj.__module__}.{name}"
                    class_dict[full_name] = obj
        except Exception as e:
            pass

    return class_dict


def _format_package_name(package_name):
    return PREFIX + package_name + SUFFIX


def get_mock_cls_name(cls) -> str:
    if isinstance(cls, type):
        cls = f"{cls.__module__}.{cls.__name__}"

    elif isinstance(cls, types.FunctionType):
        module = inspect.getmodule(cls)
        func_name = get_proxy_func_name(cls.__name__)
        cls = f"{module.__name__}.{func_name}"
        
    pkg_name, cls_ = cls.split(".", 1)
    pkg_name = _format_package_name(pkg_name)
    return f"{pkg_name}.{cls_}"

