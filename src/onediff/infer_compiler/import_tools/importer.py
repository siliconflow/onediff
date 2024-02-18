import os
import sys
import importlib
from inspect import ismodule
from typing import Optional, Union
from types import FunctionType, ModuleType
from pathlib import Path
from importlib.metadata import requires
from .format_utils import MockEntityNameFormatter
from .dyn_mock_mod import DynamicMockModule
from ..utils.log_utils import logger

__all__ = ["LazyMocker", "is_need_mock"]


def is_need_mock(cls) -> bool:
    assert isinstance(cls, (type, str))
    main_pkg = cls.__module__.split(".")[0]
    try:
        if main_pkg == "torch":
            return True
        pkgs = requires(main_pkg)
    except Exception as e:
        logger.info(f"Error when checking need mock of package {main_pkg}: {e}")
        return True
    if pkgs:
        for pkg in pkgs:
            pkg = pkg.split(" ")[0]
            if pkg == "torch":
                return True
        return False
    return True


class DynamicModuleLoader(ModuleType):
    def __init__(self, obj_entity: ModuleType, pkg_root=None, module_path=None):
        self.obj_entity = obj_entity
        self.pkg_root = pkg_root
        self.module_path = module_path

    @classmethod
    def from_path(cls, module_path: str):
        model_name = os.path.basename(module_path)
        module = importlib.import_module(model_name)
        return cls(module, module_path, module_path)

    def __getattr__(self, name):
        obj_entity = getattr(self.obj_entity, name, None)
        module_path = os.path.join(self.module_path, name)
        if obj_entity is None:
            pkg_name = os.path.basename(self.pkg_root)
            absolute_name = os.path.relpath(module_path, self.pkg_root).replace(
                os.path.sep, "."
            )
            absolute_name = f"{pkg_name}.{absolute_name}"
            obj_entity = importlib.import_module(absolute_name)
        if ismodule(obj_entity):
            return DynamicModuleLoader(obj_entity, self.pkg_root, module_path)
        return obj_entity


class LazyMocker:
    def __init__(self, prefix: str, suffix: str, tmp_dir: Optional[Union[str, Path]]):
        self.prefix = prefix
        self.suffix = suffix
        self.tmp_dir = tmp_dir
        self.mocked_packages = set()
        self.cleanup_list = []

    def mock_package(self, package: str):
        pass

    def cleanup(self):
        pass

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

    def add_mocked_package(self, package: str):
        if package in self.mocked_packages:
            return

        self.mocked_packages.add(package)
        package = sys.modules.get(package, None)

        # TODO remove code below
        # fix the mock error in https://github.com/siliconflow/oneflow/blob/main/python/oneflow/mock_torch/mock_importer.py#L105-L118
        if package and getattr(package, "__file__", None) is not None:
            pkg_path = Path(package.__file__).parents[1]
            if pkg_path not in sys.path:
                sys.path.append(str(pkg_path))

    def load_entity_with_mock(self, entity: Union[str, type, FunctionType]):
        formatter = MockEntityNameFormatter(prefix=self.prefix, suffix=self.suffix)
        full_obj_name = formatter.format(entity)
        attrs = full_obj_name.split(".")

        # add package path to sys.path to avoid mock error
        self.add_mocked_package(attrs[0])

        mock_pkg = DynamicMockModule.from_package(attrs[0], verbose=False)
        for name in attrs[1:]:
            mock_pkg = getattr(mock_pkg, name)
        return mock_pkg
