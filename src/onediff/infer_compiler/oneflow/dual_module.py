import os
import types
from typing import Any
from itertools import chain

import torch
import oneflow as flow
from oneflow.utils.tensor import to_torch

from ..transform.builtin_transform import torch2oflow
from ..utils.oneflow_exec_mode import oneflow_exec_mode, oneflow_exec_mode_enabled
from ..utils.log_utils import logger


class DualModule(torch.nn.Module):
    def __init__(self, torch_module, oneflow_module):
        torch.nn.Module.__init__(self)
        object.__setattr__(self, "_torch_module", torch_module)
        object.__setattr__(self, "_oneflow_module", oneflow_module)
        object.__setattr__(self, "_modules", torch_module._modules)
        object.__setattr__(self, "_parameters", torch_module._parameters)
        object.__setattr__(self, "_buffers", torch_module._buffers)

    @property
    def oneflow_module(self):
        if self._oneflow_module is not None:
            return self._oneflow_module

        logger.debug(f"Convert {type(self._torch_module)} ...")
        self._oneflow_module = torch2oflow(self._torch_module)
        logger.debug(f"Convert {type(self._torch_module)} done!")
        return self._oneflow_module

    @oneflow_module.deleter
    def oneflow_module(self):
        if self._oneflow_module:
            del self._oneflow_module
            setattr(self, "_oneflow_module", None)

    def to(self, *args, **kwargs):
        if oneflow_exec_mode_enabled():
            self._oneflow_module.to(*args, **kwargs)
        else:
            if self._oneflow_module is not None:
                of_args = [torch2oflow(v) for v in args]
                of_kwargs = {k: torch2oflow(v) for k, v in kwargs.items()}
                self._oneflow_module.to(*of_args, **of_kwargs)
                self._torch_module_to_with_check(*args, **kwargs)
            else:
                self._torch_module.to(*args, **kwargs)

    def _torch_module_to_with_check(self, *args, **kwargs):
        def _align_tensor(torch_module, oneflow_module):
            oneflow_tensor_list = set(
                [x for x, _ in oneflow_module.named_parameters()]
                + [x for x, _ in oneflow_module.named_buffers()]
            )
            for name, tensor in chain.from_iterable(
                [torch_module.named_parameters(), torch_module.named_buffers(),]
            ):
                if name not in oneflow_tensor_list:
                    tensor.data = tensor.to(*args, **kwargs)
                else:
                    oneflow_tensor = oneflow_module.get_parameter(name)
                    if oneflow_tensor is None:
                        tensor.data = tensor.to(*args, **kwargs)
                    elif tensor.data_ptr() != oneflow_tensor.data_ptr():
                        tensor.data = to_torch(oneflow_tensor.data)

        oneflow_module_list = set([x for x, _ in self._oneflow_module.named_modules()])
        for name, module in self._torch_module.named_modules():
            if name not in oneflow_module_list:
                module.to(*args, **kwargs)
            else:
                _align_tensor(module, self._oneflow_module.get_submodule(name))

    def __getattr__(self, name):
        if name == "_torch_module" or name == "_oneflow_module":
            return super().__getattribute__(name)

        torch_attr = getattr(self._torch_module, name)
        oneflow_attr = (
            None
            if self._oneflow_module is None
            else getattr(self._oneflow_module, name)
        )

        if isinstance(torch_attr, torch.nn.ModuleList):
            if oneflow_attr is None:
                oneflow_attr = flow.nn.ModuleList([None] * len(torch_attr))
            return DualModuleList(torch_attr, oneflow_attr)

        elif isinstance(torch_attr, torch.nn.Module):
            from .utils import get_mixed_dual_module

            return get_mixed_dual_module(torch_attr.__class__)(torch_attr, oneflow_attr)
        else:
            return oneflow_attr if oneflow_exec_mode_enabled() else torch_attr

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ["_torch_module", "_oneflow_module"]:
            super().__setattr__(name, value)
        else:  # TODO: aviod memory up when set attr
            if self._oneflow_module is not None:
                v = torch2oflow(value)
                if isinstance(v, flow.Tensor):
                    obj = getattr(self._oneflow_module, name)
                    obj.copy_(v)
                else:
                    setattr(self._oneflow_module, name, v)
            setattr(self._torch_module, name, value)

    def extra_repr(self) -> str:
        return self._torch_module.extra_repr()


class DualModuleList(torch.nn.ModuleList):
    def __init__(self, torch_modules, oneflow_modules):
        super().__init__()
        assert len(torch_modules) == len(oneflow_modules)
        self._torch_modules = torch_modules
        self._oneflow_modules = oneflow_modules
        from .utils import get_mixed_dual_module

        dual_modules = []
        for torch_module, oneflow_module in zip(
            self._torch_modules, self._oneflow_modules
        ):
            dual_modules.append(
                get_mixed_dual_module(torch_module.__class__)(
                    torch_module, oneflow_module
                )
            )
        # clear self._modules since `self._torch_modules = torch_modules` will append a module to self._modules
        self._modules.clear()
        self += dual_modules

    def __setitem__(self, idx: int, module: DualModule):
        idx = self._get_abs_string_index(idx)
        setattr(self._torch_modules, str(idx), module._torch_module)
        setattr(self._oneflow_modules, str(idx), module._oneflow_module)
        return setattr(self, str(idx), module)

    def __setattr__(self, key, value):
        if key in ("_torch_modules", "_oneflow_modules"):
            return object.__setattr__(self, key, value)
        if isinstance(value, DualModule):
            setattr(self._torch_modules, key, value._torch_module)
            setattr(self._oneflow_modules, key, value._oneflow_module)
        else:
            setattr(self._torch_modules, key, value)
            value = torch2oflow(value)
            setattr(self._oneflow_modules, key, value)
        return object.__setattr__(self, key, value)
