from inspect import ismodule, signature
from types import ModuleType
from copy import deepcopy
from contextlib import contextmanager
from typing import List, Dict
import importlib
import inspect
import os
import torch
from oneflow.mock_torch import enable
from oneflow.mock_torch.mock_importer import _importer
from .import_module_utils import import_module_from_path
from ..utils.log_utils import logger
from ..utils.patch_for_compiler import *

__all__ = ["DynamicMockModule"]


def _get_module(full_name: str):
    try:
        attrs = full_name.split(".")
        module = importlib.import_module(attrs[0])
        for attr in attrs[1:]:
            module = getattr(module, attr)
        return module
    except Exception as e:
        pass


def inspect_modules_and_attributes(module_names):
    all_results = {}

    def get_attribute_info(module_name):
        module = _get_module(module_name)
        if module is None:
            return {}
        attribute_info = {}
        for attr in (a for a in dir(module) if not a.startswith("__")):
            try:
                attr_value = getattr(module, attr)
                source_file = inspect.getsourcefile(attr_value)
                attribute_info[f"{module_name};{attr}"] = (source_file, f"{attr_value}")
            except Exception as e:
                pass

        return attribute_info

    for name in module_names:
        all_results.update(get_attribute_info(name))

    return all_results


def getattr_from_module_name(module, module_name: str):
    # case <function get_additional_models_factory.<locals>.get_additional_models_with_motion at 0x7fd0542cf6d0>
    full_attr_name = module_name.split(" ")[1]
    attrs = full_attr_name.split(".")
    sub_module = module
    for attr in attrs:
        if attr == "<locals>":
            if len(signature(sub_module).parameters) == 0:
                logger.info(f"{full_attr_name=} is a local function without parameters")

                def proxy_func(*args, **kwargs):
                    return sub_module()(*args, **kwargs)

                return proxy_func
            else:
                # logger.warning(
                #     f"Not support {module_name} with parameters Module: {module}"
                # )
                raise RuntimeError(
                    f"Not support {module_name} with parameters Module: {module}"
                )
        sub_module = getattr(sub_module, attr, None)
    return sub_module


def _update_module(full_names, main_pkg_enable_context):
    with main_pkg_enable_context():
        original_results = inspect_modules_and_attributes(full_names)

    updated_results = inspect_modules_and_attributes(full_names)

    torch_path = os.path.dirname(torch.__file__)

    for module_key, (module_path, module_code) in updated_results.items():
        org_module_path, org_module_code = original_results.get(
            module_key, (module_path, module_code)
        )

        if org_module_path != module_path or org_module_code != module_code:
            # Skip torch module Because torch module is already mocked by oneflow
            if torch_path in module_path:
                continue

            # Update module inplace
            module_name, attr_name = module_key.split(";")
            # sample_module = importlib.import_module(module_name).__dict__[attr_name]
            good_module = _get_module(module_name)
            sample_module = getattr(good_module, attr_name)
            package_space = inspect.getmodule(sample_module).__name__
            with main_pkg_enable_context():
                module = _get_module(module_name)
                if package_space == "__main__":
                    other = import_module_from_path(module_path)
                else:
                    other = importlib.import_module(package_space)

                value = getattr_from_module_name(other, module_name=str(sample_module))
                if value is None:
                    continue

                setattr(module, attr_name, value)


class DynamicMockModule(ModuleType):
    def __init__(
        self, pkg_name: str, obj_entity: ModuleType, main_pkg_enable: callable,
    ):
        self._pkg_name = pkg_name
        self._obj_entity = obj_entity  # ModuleType or _LazyModule
        self._main_pkg_enable = main_pkg_enable
        self._intercept_dict = {}

    def __repr__(self) -> str:
        return f"<DynamicMockModule {self._pkg_name} {self._obj_entity}>"

    def hijack(self, module_name: str, obj: object):
        self._intercept_dict[module_name] = obj

    @classmethod
    def from_package(
        cls,
        main_pkg: str,
        *,
        lazy: bool = True,
        verbose: bool = False,
        extra_dict: Dict[str, str] = None,
        required_dependencies: List[str] = [],
    ):
        assert isinstance(main_pkg, str)

        @contextmanager
        def main_pkg_enable():
            with enable(
                lazy=lazy,
                verbose=verbose,
                extra_dict=extra_dict,
                main_pkg=main_pkg,
                mock_version=True,
                required_dependencies=required_dependencies,
            ):
                yield

        with main_pkg_enable():
            obj_entity = importlib.import_module(main_pkg)
        return cls(main_pkg, obj_entity, main_pkg_enable)

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
        fullname = f"{self._obj_entity.__name__}.{name}"
        if fullname in self._intercept_dict:
            return self._intercept_dict[fullname]

        with self._main_pkg_enable():
            obj_entity = getattr(self._obj_entity, name, None)
            if obj_entity is None:
                obj_entity = self._get_module(name)
            org_delete_list = deepcopy(_importer.delete_list)

        try:
            # Update obj_entity inplace
            if not _importer.enable:
                _update_module([fullname] + org_delete_list, self._main_pkg_enable)
        except Exception as e:
            logger.debug(f"Failed to update obj_entity in place. Exception: {e}")

        if ismodule(obj_entity):
            return DynamicMockModule(self._pkg_name, obj_entity, self._main_pkg_enable)

        return obj_entity

    def __all__(self):
        with self._main_pkg_enable():
            return dir(self._obj_entity)
