import inspect
import os
import sys
import importlib
from typing import Optional, Union
from types import FunctionType, ModuleType
from pathlib import Path
from .copier import PackageCopier
from .context_managers import onediff_mock_torch


PREFIX = "mock_"
SUFFIX = f"_oflow_{os.getpid()}"

__all__ = [
    "import_module_from_path",
    "copy_package",
    "get_mock_entity_name",
    "load_entity_with_mock",
]


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


def copy_package(package: str, target_directory: Optional[Union[str, Path]] = None):
    """Copy package to target directory"""
    with onediff_mock_torch():
        copier = PackageCopier(
            package, prefix=PREFIX, suffix=SUFFIX, target_directory=target_directory
        )
        copier.do()  # copy package


def _format_package_name(package_name: str) -> str:
    if package_name.startswith(PREFIX) and package_name.endswith(SUFFIX):
        return package_name
    return PREFIX + package_name + SUFFIX


def _format_full_class_name(obj: Union[str, type, FunctionType]):

    if isinstance(obj, type):
        obj = f"{obj.__module__}.{obj.__name__}"

    elif isinstance(obj, FunctionType):
        module = inspect.getmodule(obj)
        obj = f"{module.__name__}.{obj.__name__}"

    assert isinstance(obj, str), f"obj must be str, but got {type(obj)}"
    pkg_name, cls_name = obj.split(".", 1)
    pkg_name = _format_package_name(pkg_name)
    return f"{pkg_name}.{cls_name}"


def get_mock_entity_name(obj: Union[str, type, FunctionType]) -> str:
    full_obj_name = _format_full_class_name(obj)
    return full_obj_name


def load_entity_with_mock(obj: Union[str, type, FunctionType]):
    path = _format_full_class_name(obj)
    attrs = path.split(".")
    with onediff_mock_torch():
        obj_entity = importlib.reload(sys.modules[attrs[0]])
        for name in attrs[1:]:
            obj_entity = getattr(obj_entity, name)
        return obj_entity
