import pkgutil
import importlib
import inspect
from typing import Dict

def import_submodules(package, recursive=True):
    """ Import all submodules of a module, recursively, including subpackages """
    if isinstance(package, str):
        package = importlib.import_module(package)
    
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        try:
            good_import = importlib.import_module(full_name)
            yield good_import
        except ImportError as e:
            print(f"Failed to import {full_name}: {e}")

        if recursive and is_pkg:
            yield from import_submodules(full_name)

def get_classes_in_package(package, base_class=None)->Dict[str,object]:
    """
    Get all classes in a package and its submodules.

    Args:
        package (str or module): The package to search for classes.
        base_class (type, optional): The base class to filter classes by.

    Returns:
        dict: A dictionary mapping full class names to class objects.
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    
    class_dict = {}
    
    for module in import_submodules(package):
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if inspect.isclass(obj) and (base_class is None or issubclass(obj, base_class)):
                full_name = f'{obj.__module__}.{name}'
                class_dict[full_name] = obj 
    
    return class_dict



if __name__ == "__main__":
    package_name = 'diffusers.models'
    base_class = None  # Change this to your desired base class

    class_dict = get_classes_in_package(package_name, base_class)

    for full_name, cls in class_dict.items():
        print(f'{full_name}')

