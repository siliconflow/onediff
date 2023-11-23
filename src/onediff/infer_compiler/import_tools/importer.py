import inspect
import os
import sys
import time
import importlib
from typing import Optional, Union
from types import FunctionType, ModuleType
from pathlib import Path
from .copier import PackageCopier
from .context_managers import onediff_mock_torch, cache_package

__all__ = [
    "import_module_from_path",
    "mock_package",
    "get_mock_entity_name",
    "load_entity_with_mock",
]


def gen_unique_id():
    timestamp = int(time.time() * 1000)
    process_id = os.getpid()
    # TODO(): refine the unique id
    # sequence = str(uuid.uuid4())
    unique_id = f"{timestamp}{process_id}"
    return unique_id

PREFIX = "mock_"
SUFFIX = f"_oflow_{gen_unique_id()}"


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


def mock_package(package: str, output_directory: Optional[Union[str, Path]] = None):
    """Mock the package and cache the pkg ModuleType object.""" 
    with onediff_mock_torch(package):
        copier = PackageCopier(
            package, prefix=PREFIX, suffix=SUFFIX, target_directory=output_directory
        )
        copier.mock()  # Mock the package.
        new_pkg_name = copier.new_pkg_name
        mock_pkg = copier.get_import_module()
        copier.cleanup()  # remove copied package
        cache_package(new_pkg_name, mock_pkg)
        return mock_pkg


def _format_pkg_name(pkg_name: str) -> str:
    if pkg_name.startswith(PREFIX) and pkg_name.endswith(SUFFIX):
        return pkg_name
    return PREFIX + pkg_name + SUFFIX

def _reverse_pkg_name(pkg_name: str) -> str:
    assert pkg_name.startswith(PREFIX) and pkg_name.endswith(SUFFIX), \
        f"Package name must start with {PREFIX} and end with {SUFFIX}, but got {pkg_name}"
    return pkg_name[len(PREFIX):-len(SUFFIX)]

def _format_full_class_name(obj: Union[str, type, FunctionType]):
    if isinstance(obj, type):
        obj = f"{obj.__module__}.{obj.__name__}"

    elif isinstance(obj, FunctionType):
        module = inspect.getmodule(obj)
        obj = f"{module.__name__}.{obj.__name__}"

    assert isinstance(obj, str), f"obj must be str, but got {type(obj)}"
    
    if '.' in obj:
        pkg_name, cls_name = obj.split('.', 1)
        return f"{_format_pkg_name(pkg_name)}.{cls_name}"
    else:
        return _format_pkg_name(obj)

def get_mock_entity_name(entity: Union[str, type, FunctionType]) -> str:
    full_obj_name = _format_full_class_name(entity)
    return full_obj_name
    

def load_entity_with_mock(entity: Union[str, type, FunctionType],
                           *,
                        output_directory: Optional[Union[str, Path]] = None):
    """
    Load entity with mock support. If specified (`entity`) not found, mock its package and retry.

    Args:
        `entity`: The entity to be loaded.

        `output_directory`: The directory to store the mock package. 

                            Defaults to the current directory (".").

    Example 1:
    >>> import module
    >>> cls = module.Class  
    >>> mock_entity = load_entity_with_mock(cls)  

    Example 2:
    >>> full_cls_name = “module.Class”
    >>> mock_entity = load_entity_with_mock(full_cls_name)
    """
    full_obj_name = _format_full_class_name(entity)
    attrs = full_obj_name.split(".")
    with onediff_mock_torch(attrs[0]) as obj_entity:
        if obj_entity is not None:
            for name in attrs[1:]:
                obj_entity = getattr(obj_entity, name)
            return obj_entity
        
    pkg_name = _reverse_pkg_name(attrs[0])
    pkg = importlib.import_module(pkg_name)
    if pkg is None:    
        raise ValueError(f"package {pkg_name} not found in sys.modules")
    
    # https://docs.python.org/3/reference/import.html#path__
    # Do something with the output directory
    mock_package(pkg.__path__[0], output_directory=output_directory)
    return load_entity_with_mock(entity)

