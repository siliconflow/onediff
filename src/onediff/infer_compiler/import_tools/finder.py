import inspect
import pkgutil
import importlib
from typing import Dict
from .copier import PackageCopier
from .printer import print_red


PREFIX = "mock_"
SUFFIX = ""



def import_submodules(package, recursive=True):
    """Import all submodules of a module, recursively, including subpackages"""
    if isinstance(package, str):
        package = importlib.import_module(package)

    for _, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        try:
            # TODO(): Avoid copy, rename comfy.x.x.x with mocked_comfy.x.x.x
            good_import = importlib.import_module(full_name)
            yield good_import

        except Exception as e:
            # logger.debug(f"Failed to import {full_name}: {e}")
            pass # ignore
            

        if recursive and is_pkg:
            try:
                yield from import_submodules(full_name)
            except Exception as e:
                pass # ignore


def get_classes_in_package(package, base_class=None) -> Dict[str, type]:
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


def get_mock_cls_name(cls)->str:
    if isinstance(cls, type):
        cls = f"{cls.__module__}.{cls.__name__}"
        
    return PREFIX + cls + SUFFIX






