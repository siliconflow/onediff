from functools import wraps

from ..transform.builtin_transform import torch2oflow
from ..transform.manager import transform_mgr
from ..utils.log_utils import logger
from .dual_module import DualModule


@torch2oflow.register
def _(mod: DualModule, verbose=False):
    return torch2oflow(mod._torch_module, verbose)


def handle_deployable_exception(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if transform_mgr.debug_mode:
            return func(self, *args, **kwargs)
        else:
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e=}")
                logger.warning("Recompile oneflow module ...")
                del self._deployable_module_model.oneflow_module
                self._deployable_module_dpl_graph = None
                return func(self, *args, **kwargs)

    return wrapper


def get_mixed_dual_module(module_cls):
    if issubclass(module_cls, DualModule) and "MixedDualModule" in module_cls.__name__:
        return module_cls

    class MixedDualModule(DualModule, module_cls):
        def __init__(self, torch_module, oneflow_module):
            while isinstance(torch_module, DualModule):
                torch_module = torch_module._torch_module
            DualModule.__init__(self, torch_module, oneflow_module)

        def _get_name(self) -> str:
            return f"{self.__class__.__name__}(of {module_cls.__name__})"

    return MixedDualModule


# Return a OneflowDeployableModule that using module_cls as it's parent class.
def get_mixed_deployable_module(module_cls):
    from .deployable_module import OneflowDeployableModule

    class MixedOneflowDeployableModule(OneflowDeployableModule, module_cls):
        def __init__(self, torch_module, oneflow_module, dynamic=True, options=None):
            OneflowDeployableModule.__init__(
                self, torch_module, oneflow_module, dynamic, options
            )
            self._is_raw_deployable_module = False

        @classmethod
        def from_existing(cls, existing_module, dynamic=True, options=None):
            torch_module = existing_module._deployable_module_model._torch_module
            oneflow_module = existing_module._deployable_module_model._oneflow_module
            instance = cls(torch_module, oneflow_module, dynamic, options)
            instance._deployable_module_dpl_graph = None
            if hasattr(existing_module, "_deployable_module_dpl_graph"):
                instance._deployable_module_dpl_graph = (
                    existing_module._deployable_module_dpl_graph
                )
            return instance

        def _get_name(self):
            return f"{self.__class__.__name__}(of {module_cls.__name__})"

    return MixedOneflowDeployableModule


def get_oneflow_graph(model, size=9, dynamic_graph=True):
    from .graph import OneflowGraph

    g = OneflowGraph(model)
    g._dynamic_input_graph_cache.set_cache_size(size)
    g._dynamic_input_graph_cache.enable_shared(dynamic_graph)
    return g
