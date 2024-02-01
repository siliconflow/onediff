import torch
import torch.nn as nn
from types import MethodType
from .transform.builtin_transform import torch2oflow
from .utils.args_tree_util import process_input, process_output
from .utils.oneflow_exec_mode import oneflow_exec_mode
from .utils.log_utils import logger
from .with_oneflow_compile import get_oneflow_graph
from .transform.manager import transform_mgr


def default_is_leaf_fn(attr):
    if isinstance(attr, torch.dtype):
        return True
    if isinstance(attr, MethodType):
        return True
    return False



class OneFlowModule:
    def __init__(self, use_graph=True, dynamic=True, options={}):
        self.of_module = None
        self.use_graph = use_graph
        self.dynamic = dynamic
        self.options = options
        self.module_dpl_graph = None
        self.current_input_count = None

    @property
    def compiled_model(self):
        if not self.is_compiled():
            return None
        if self.use_graph:
            return self.module_dpl_graph
        else:
            return self.of_module

    def is_compiled(self):
        return self.of_module is not None

    def _validate(self, pt_module: nn.Module):
        if not self.is_compiled():
            logger.warning("OneFlowModule has not been compiled, please compile first")
            return False

        # check state_dict keys equal
        state_dict = pt_module.state_dict().keys()
        of_state_dict = self.of_module.state_dict().keys()
        assert set(state_dict) == set(of_state_dict)
        return True

    def update_module(self, key, value):
        # TODO support weight update
        raise NotImplementedError

    def compile(self, pt_module: nn.Module):
        assert isinstance(pt_module, nn.Module)
        if self.is_compiled():
            logger.warning(
                "OneFlowModule has been compiled, recompile will overwrite the previous compiled module"
            )
            del self.module_dpl_graph
        self.of_module = torch2oflow(pt_module)

        assert self._validate(pt_module)

        if self.use_graph:
            logger.info("OneFlowModule Use Graph Mode")
            self.module_dpl_graph = get_oneflow_graph(
                self.of_module, self.options.get("size", 9), self.dynamic
            )
            if transform_mgr.debug_mode:
                self.module_dpl_graph.debug(self.options.get("debug", 0))
        else:
            logger.info("OneFlowModule Use Eager Mode")

    def __call__(self, *args, **kwargs):
        mapped_args, mapped_kwargs, _ = process_input(*args, **kwargs)
        with oneflow_exec_mode(self.use_graph):
            out =  self.compiled_model(*mapped_args, **mapped_kwargs)
        return process_output(out)
        
    
class Proxy:
    __attrs = ["_proxy", "_is_leaf_fn", "_proxy_of"]

    @property
    def __class__(self):
        return self._proxy.__class__

    def __init__(self, pt_module, is_leaf_fn=default_is_leaf_fn):
        self._proxy = pt_module
        self._is_leaf_fn = is_leaf_fn
        self._proxy_of = OneFlowModule(use_graph=True, dynamic=False)

    def __getattr__(self, name):
        if name in Proxy.__attrs:
            return object.__getattribute__(self, name)

        attr = getattr(self._proxy, name)

        if self._is_leaf_fn(attr):
            return attr
        else:
            return Proxy(attr, self._is_leaf_fn)

    def __call__(self, *args, **kwargs):
        if not self._proxy_of.is_compiled():
            self._proxy_of.compile(self._proxy)
        return self._proxy_of(*args, **kwargs)

    def __setitem__(self, key, value):
        self._proxy[key] = value

    def __getitem__(self, key):
        # TODO support Proxy(self._proxy[key])
        return self._proxy[key]

    def __setattr__(self, name, value):
        if name in Proxy.__attrs:
            object.__setattr__(self, name, value)
        else:
            setattr(self._proxy, name, value)


if __name__ == "__main__":
    model = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU())
    proxy = Proxy(model)
    print(proxy.state_dict().keys() == model.state_dict().keys())
    print(isinstance(model, nn.Sequential))
    print(isinstance(proxy, nn.Sequential))
    print(proxy.__class__ == model.__class__)
