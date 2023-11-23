import inspect
import os
import sys
import time
import importlib
from typing import Optional, Union
from types import FunctionType, ModuleType
from pathlib import Path
from .copier import PackageCopier
from .context_managers import onediff_mock_torch
from .format_utils import MockEntityNameFormatter

__all__ = ["import_module_from_path", "LazyMocker"]


class MockEntity:
    def __init__(self, obj_entity: Optional[Union[type, ModuleType]] = None):
        self._obj_entity = obj_entity

    @classmethod
    def from_package(cls, package: str):
        with onediff_mock_torch():
            return cls(importlib.import_module(package))

    def __getattr__(self, name: str):
        with onediff_mock_torch():
            try:
                obj_entity = getattr(self._obj_entity, name)
            except AttributeError:
                # Fix Lazy import
                # https://github.com/huggingface/diffusers/blob/main/src/diffusers/__init__.py#L728-L734
                obj_entity = importlib.import_module(
                    f"{self._obj_entity.__name__}.{name}"
                )
                if obj_entity is None:
                    raise ValueError(
                        f"Attribute {name} not found in {self._obj_entity}"
                    )

            if inspect.ismodule(obj_entity):
                return MockEntity(obj_entity)
            return obj_entity

    def entity(self):
        return self._obj_entity


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


class LazyMocker:
    def __init__(self, prefix: str, suffix: str, tmp_dir: Optional[Union[str, Path]]):
        self.prefix = prefix
        self.suffix = suffix
        self.tmp_dir = tmp_dir
        self.mocked_packages = set()

    def mock_package(self, package: str):
        # TODO Mock the package in memory
        with onediff_mock_torch():
            copier = PackageCopier(
                package,
                prefix=self.prefix,
                suffix=self.suffix,
                target_directory=self.tmp_dir,
            )
            copier.mock()
        self.mocked_packages.add(package)

    def get_mock_entity_name(self, entity: Union[str, type, FunctionType]):
        formatter = MockEntityNameFormatter(prefix=self.prefix, suffix=self.suffix)
        full_obj_name = formatter.format(entity)
        return full_obj_name

    def mock_entity(self, entity: Union[str, type, FunctionType]):
        return self.load_entity_with_mock(entity)

    def load_entity_with_mock(self, entity: Union[str, type, FunctionType]):
        formatter = MockEntityNameFormatter(prefix=self.prefix, suffix=self.suffix)
        full_obj_name = formatter.format(entity)
        attrs = full_obj_name.split(".")
        try:
            obj_entity = MockEntity.from_package(attrs[0])
            for name in attrs[1:]:
                obj_entity = getattr(obj_entity, name)
            return obj_entity
        except ModuleNotFoundError:
            pkg_name = formatter.unformat(attrs[0])
            pkg = importlib.import_module(pkg_name)
            if pkg is None:
                raise ValueError(f"package {pkg_name} not found in sys.modules")
            # https://docs.python.org/3/reference/import.html#path__
            self.mock_package(pkg.__path__[0])
            return self.load_entity_with_mock(entity)

        except Exception as e:
            raise e
