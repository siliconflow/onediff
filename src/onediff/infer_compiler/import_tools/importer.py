import inspect
import os
import sys
import importlib
import shutil
from typing import Any, Optional, Union
from types import FunctionType, ModuleType
from pathlib import Path
from ..utils.log_utils import logger
from .copier import PackageCopier
from .mock_torch_context import onediff_mock_torch
from .format_utils import MockEntityNameFormatter

__all__ = ["import_module_from_path", "LazyMocker"]


class MockEntity:
    def __init__(self, obj_entity: ModuleType = None):
        self._obj_entity = obj_entity  # ModuleType or _LazyModule

    @classmethod
    def from_package(cls, package: str):
        with onediff_mock_torch():
            return cls(importlib.import_module(package))

    def _get_module(self, _name: str):
        # Fix Lazy import
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/__init__.py#L728-L734
        module_name = f"{self._obj_entity.__name__}.{_name}"
        try:
            return importlib.import_module(module_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __getattr__(self, name: str):
        with onediff_mock_torch():
            obj_entity = getattr(self._obj_entity, name, None)
            if obj_entity is None:
                obj_entity = self._get_module(name)

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
        self.cleanup_list = []

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
            self.mocked_packages.add(copier.new_pkg_name)
            self.cleanup_list.append(copier.new_pkg_path)

    def cleanup(self):
        for path in self.cleanup_list:
            logger.debug(f"Removing {path=}")
            shutil.rmtree(path, ignore_errors=True)

    def get_mock_entity_name(self, entity: Union[str, type, FunctionType]):
        formatter = MockEntityNameFormatter(prefix=self.prefix, suffix=self.suffix)
        full_obj_name = formatter.format(entity)
        return full_obj_name

    def mock_entity(self, entity: Union[str, type, FunctionType]):
        """Mock the entity and return the mocked entity
        
        Example:
            >>> mocker = LazyMocker(prefix="mock_", suffix="_of", tmp_dir="tmp")
            >>> mocker.mock_entity("models.DemoModel")
            <class 'mock_models_of.DemoModel'>
            >>> cls_obj = models.DemoModel
            >>> mocker.mock_entity(cls_obj)
            <class 'mock_models_of.DemoModel'>
        """
        return self.load_entity_with_mock(entity)

    def load_entity_with_mock(self, entity: Union[str, type, FunctionType]):
        formatter = MockEntityNameFormatter(prefix=self.prefix, suffix=self.suffix)
        full_obj_name = formatter.format(entity)
        attrs = full_obj_name.split(".")
        if attrs[0] in self.mocked_packages:
            obj_entity = MockEntity.from_package(attrs[0])
            for name in attrs[1:]:
                obj_entity = getattr(obj_entity, name)
            return obj_entity
        else:
            pkg_name = formatter.unformat(attrs[0])
            pkg = importlib.import_module(pkg_name)
            if pkg is None:
                RuntimeError(f'Importing package "{pkg_name}" failed')
            # https://docs.python.org/3/reference/import.html#path__
            self.mock_package(pkg.__path__[0])
            return self.load_entity_with_mock(entity)
