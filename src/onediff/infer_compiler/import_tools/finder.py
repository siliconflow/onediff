import pkgutil
import importlib
import inspect
import logging
from typing import Dict
from .copier import PackageCopier

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

PREFIX = "mock_"
SUFFIX = ""


def import_submodules(package, recursive=True):
    """Import all submodules of a module, recursively, including subpackages"""
    if isinstance(package, str):
        package = importlib.import_module(package)

    for _, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        try:
            good_import = importlib.import_module(full_name)
            yield good_import

        except ImportError as e:
            logger.warning(f"Failed to import {full_name}: {e}")

        if recursive and is_pkg:
            yield from import_submodules(full_name)


def get_classes_in_package(package, base_class=None) -> Dict[str, object]:
    """
    Get all classes in a package and its submodules.

    Args:
        package (str or module): The package to search for classes.
        base_class (type, optional): The base class to filter classes by.

    Returns:
        dict: A dictionary mapping full class names to class objects.
    """
    if isinstance(package, str):
        copier = PackageCopier(package, prefix=PREFIX, suffix=SUFFIX)
        copier()  # copy package
        package = copier.get_import_module()

    class_dict = {}

    for module in import_submodules(package):
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if inspect.isclass(obj) and (
                base_class is None or issubclass(obj, base_class)
            ):
                full_name = f"{obj.__module__}.{name}"
                class_dict[full_name] = obj

    return class_dict


def get_mock_cls_name(full_cls_name):
    return PREFIX + full_cls_name + SUFFIX
