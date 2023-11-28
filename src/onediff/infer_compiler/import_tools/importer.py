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
        self.formatter = MockEntityNameFormatter(prefix=self.prefix, suffix=self.suffix)
        self.tmp_dir = tmp_dir
        self.mocked_packages = {}
        self.cleanup_list = []

    def mock_package(self, package: str):
        # TODO Mock the package in memory
        copier = PackageCopier(
            package,
            prefix=self.prefix,
            suffix=self.suffix,
            target_directory=self.tmp_dir,
        )
        mocked_pkg = copier.mock()
        self.mocked_packages[copier.new_pkg_name] = mocked_pkg
        self.cleanup_list.append(copier.new_pkg_path)
        return mocked_pkg

    def cleanup(self):
        for path in self.cleanup_list:
            logger.debug(f"Removing {path=}")
            shutil.rmtree(path, ignore_errors=True)

    def get_mock_entity_name(self, entity: Union[str, type, FunctionType]):
        full_obj_name = self.formatter.format(entity)
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
        full_obj_name = self.formatter.format(entity)
        attrs = full_obj_name.split(".")
        mocked_pkg_name = attrs[0]
        if mocked_pkg_name not in self.mocked_packages:
            pkg_name = self.formatter.unformat(attrs[0])
            pkg = importlib.import_module(pkg_name)
            if pkg is None:
                RuntimeError(f'Importing package "{pkg_name}" failed')
            # https://docs.python.org/3/reference/import.html#path__
            obj_entity = self.mock_package(pkg.__path__[0])
        else:
            obj_entity = self.mocked_packages[mocked_pkg_name]
        
        for attr in attrs[1:]:
            if hasattr(obj_entity, attr):
                obj_entity = getattr(obj_entity, attr)
            else:
                # import with mock
                module_name = f"{obj_entity.__name__}.{attr}"
                try:
                    with onediff_mock_torch():
                        cur_obj = importlib.import_module(module_name)
                    setattr(obj_entity, attr, cur_obj)
                    obj_entity = cur_obj
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to import {module_name} because of the following error (look up to see its"
                        f" traceback):\n{e}"
                    ) from e
        return obj_entity
