import os
import sys
import importlib
from typing import Optional, Union
from types import FunctionType, ModuleType
from oneflow.mock_torch import DynamicMockModule
from pathlib import Path
from importlib.metadata import requires
from .format_utils import MockEntityNameFormatter

__all__ = ["import_module_from_path", "LazyMocker", "is_need_mock"]


def is_need_mock(cls) -> bool:
    assert isinstance(cls, (type, str))
    main_pkg = cls.__module__.split(".")[0]
    try:
        pkgs = requires(main_pkg)
    except Exception as e:
        return True
    if pkgs:
        for pkg in pkgs:
            pkg = pkg.split(" ")[0]
            if pkg == "torch":
                return True
        return False
    return True


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

    def load_entity_with_mock(self, entity: Union[str, type, FunctionType]):
        formatter = MockEntityNameFormatter(prefix=self.prefix, suffix=self.suffix)
        full_obj_name = formatter.format(entity)
        attrs = full_obj_name.split(".")
        self.mocked_packages.add(attrs[0])

        try:
            attrs_0_pkg = sys.modules.get(attrs[0], None)
            if attrs_0_pkg is not None:
                # add package path to sys.path to avoid mock error
                pkg_path = Path(attrs_0_pkg.__file__).parents[1]
                if pkg_path not in sys.path:
                    sys.path.append(pkg_path)
        except Exception as e:
            print(e)

        mock_pkg = DynamicMockModule.from_package(attrs[0], verbose=False)
        for name in attrs[1:]:
            mock_pkg = getattr(mock_pkg, name)
        return mock_pkg
