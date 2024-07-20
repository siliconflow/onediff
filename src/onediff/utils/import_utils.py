import importlib
import traceback
from inspect import ismodule
import os
import platform
from types import ModuleType

system = platform.system()


def check_module_availability(module_name):
    spec = importlib.util.find_spec(module_name)

    if spec:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            print(traceback.format_exc())
            return False
    else:
        return False

    return True


_oneflow_available = check_module_availability("oneflow")
_onediff_quant_available = check_module_availability("onediff_quant")
_nexfort_available = check_module_availability("nexfort")

if system != "Linux":
    print(f"Warning: OneFlow is only supported on Linux. Current system: {system}")
    _oneflow_available = False


def is_oneflow_available():
    return _oneflow_available


def is_onediff_quant_available():
    return _onediff_quant_available


def is_nexfort_available():
    return _nexfort_available


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
