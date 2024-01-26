import torch
import torch.nn as nn
from types import MethodType
import logging
from .transform.builtin_transform import torch2oflow
from .utils.args_tree_util import input_output_processor
from .utils.oneflow_exec_mode import oneflow_exec_mode
from .utils.module_operations import get_sub_module, modify_sub_module
from .utils.log_utils import logger


def default_is_leaf_fn(attr):
    if isinstance(attr, torch.dtype):
        return True
    if isinstance(attr, MethodType):
        return True
    return False


class Proxy:
    @property
    def __class__(self):
        return self._proxy.__class__

    def __init__(self, proxy_pt, is_leaf_fn=default_is_leaf_fn):
        self._proxy = proxy_pt
        self._is_leaf_fn = is_leaf_fn
        self._proxy_of = None

    def __getattr__(self, name):
        if name == "_proxy" or name == "_is_leaf_fn" or name == "_proxy_of":
            return object.__getattribute__(self, name)

        attr = getattr(self._proxy, name)
        debug_msg = (
            f"Proxy.__getattr__ {name} {attr.__class__} {self._is_leaf_fn(attr)}"
        )
        logger.debug(debug_msg)

        if self._is_leaf_fn(attr):
            return attr
        else:
            return Proxy(attr, self._is_leaf_fn)

    @input_output_processor
    def __call__(self, *args, **kwargs):
        if self._proxy_of is None:
            self._proxy_of = torch2oflow(self._proxy)
            # validate state_dict().keys()

            if self._proxy_of is None:
                raise RuntimeError("Proxy.__call__ torch2oflow failed")
            logger.info("pass torch2oflow")
            of_state_dict_keys = set(self._proxy_of.state_dict().keys())
            pt_state_dict_keys = set(self._proxy.state_dict().keys())

            sub_keys = of_state_dict_keys - pt_state_dict_keys
            if len(sub_keys) > 0:
                raise RuntimeError(
                    f"Proxy.__call__ state_dict keys not match {of_state_dict_keys} {pt_state_dict_keys}"
                )
            logger.info("pass state_dict keys check")

        _proxy_of = self._proxy_of
        with oneflow_exec_mode():
            return _proxy_of(*args, **kwargs)

    def __setitem__(self, key, value):
        self._proxy[key] = value

    def __getitem__(self, key):
        ret_pt = self._proxy[key]
        return ret_pt

    def __setattr__(self, name, value):
        if name == "_proxy" or name == "_is_leaf_fn" or name == "_proxy_of":
            object.__setattr__(self, name, value)
        else:
            debug_msg = f"Proxy.__setattr__ {name} {value.__class__}"
            logger.debug(debug_msg)
            setattr(self._proxy, name, value)


if __name__ == "__main__":
    model = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU())
    proxy = Proxy(model)
    print(proxy.state_dict().keys() == model.state_dict().keys())
    print(isinstance(model, nn.Sequential))
    print(isinstance(proxy, nn.Sequential))
    print(proxy.__class__ == model.__class__)
