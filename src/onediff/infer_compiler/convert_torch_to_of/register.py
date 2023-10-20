""" Desc: register function for torch2of
Usage:
    >>> import torch
    >>> from onediff.infer_compiler.convert_torch_to_of import torch2of
    >>> x = torch.nn.Linear(3, 4)
    >>> y = torch2of(x) # convert torch.nn.Linear to oneflow.nn.Linear
    >>> y
    <class 'oneflow.nn.modules.linear.Linear'>(in_features=3, out_features=4, bias=True)
    
### Support: 
#### Basic:(register.py)
#### Advanced:(custom_register.py)
"""
import importlib
import types
import torch
import oneflow as flow
from typing import Union
from collections import OrderedDict
from functools import singledispatch
from ..import_tools import print_red, print_yellow
from .proxy import ProxySubmodule, proxy_class
from ._globals import _WARNING_MSG

__all__ = ["torch2of", "default_converter"]


@singledispatch
def torch2of(mod, *args, **kwargs):
    global _WARNING_MSG

    msg = (
        f"Warning: No torch2of conversion interface found for: {type(mod)=}, "
        f"Default attribute retrieval method will be used. \n"
        f"You can register {type(mod)} a  conversion method in custom_register.py to suppress this warning."
    )

    if type(mod) not in _WARNING_MSG and torch2of.registry.get(type(mod), None) is None:
        print_yellow(msg)
        _WARNING_MSG.add(type(mod))

    return default_converter(mod, *args, **kwargs)


@torch.no_grad()
def default_converter(obj, verbose=False, *, proxy_cls=None):
    """Convert torch object to oneflow object."""
    try:
        new_obj_cls = proxy_class(type(obj)) if proxy_cls is None else proxy_cls

        def init(self):
            for k, v in obj.__dict__.items():
                attr = getattr(obj, k)
                self.__dict__[k] = torch2of(attr)

        of_obj_cls = type(str(new_obj_cls), (new_obj_cls,), {"__init__": init})
        of_obj = of_obj_cls()

        if verbose:
            print(f"convert {type(obj)} to {type(of_obj)}")
        return of_obj
    except Exception as e:
        # raise NotImplementedError(f"Unsupported type: {type(obj)}")
        print(f"Unsupported type: {type(obj)}")
        return obj


from .custom_register import *  # noqa: F401,F403


@torch2of.register
def _(mod: torch.nn.Module, verbose=False):
    proxy_md = ProxySubmodule(mod)

    new_md_cls = proxy_class(type(mod))

    def init(self):
        nonlocal proxy_md

        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._modules = OrderedDict()
        for (n, p) in list(proxy_md.named_parameters("", False)):
            self._parameters[n] = flow.utils.tensor.from_torch(p.data)
        for (n, b) in list(proxy_md.named_buffers("", False)):
            self._buffers[n] = flow.utils.tensor.from_torch(b.data)
        for (n, m) in proxy_md._modules.items():
            self._modules[n] = torch2of(m)

        for k, v in proxy_md.__dict__.items():
            if k not in self.__dict__:
                attr = getattr(proxy_md, k)
                try:
                    self.__dict__[k] = torch2of(attr)
                except Exception as e:
                    print_red(f"convert {type(attr)} failed: {e}")
                    raise NotImplementedError(f"Unsupported type: {type(attr)}")

    def proxy_getattr(self, attr):
        nonlocal proxy_md

        if attr in ["_parameters", "_buffers", "_modules"]:
            raise ValueError(f"missing attr {attr} in base class")
        else:
            return getattr(proxy_md, attr)

    of_mod_cls = type(
        str(new_md_cls), (new_md_cls,), {"__init__": init, "__getattr__": proxy_getattr}
    )
    of_mod = of_mod_cls()
    if of_mod.training:
        of_mod.training = False
        if verbose:
            print(
                f"warning: {type(of_mod)} is in training mode and is turned into eval mode which is good for infrence optimation."
            )

    if verbose:
        print(f"convert {type(mod)} to {type(of_mod)}")

    return of_mod


@torch2of.register
def _(mod: torch.nn.ModuleList, verbose=False):
    of_mod_list = flow.nn.ModuleList()
    for original_submod in mod:
        submod = torch2of(original_submod, verbose)
        of_mod_list.append(submod)

    return of_mod_list


@torch2of.register
def _(mod: torch.nn.Sequential, verbose=False):

    of_mod_list = []
    for original_submod in mod:
        submod = torch2of(original_submod, verbose)
        of_mod_list.append(submod)
    of_mod_seq = flow.nn.Sequential(*of_mod_list)

    return of_mod_seq


@torch2of.register
def _(mod: torch.nn.parameter.Parameter, verbose=False):
    # TODO(oneflow): assert a.requires_grad == False
    return flow.utils.tensor.from_torch(mod.data)


@torch2of.register
def _(mod: torch.Tensor, verbose=False) -> flow.Tensor:
    return flow.utils.tensor.from_torch(mod)


# torch.dtype
@torch2of.register
def _(mod: torch.dtype, verbose=False) -> flow.dtype:
    return {
        "torch.float16": flow.float16,
        "torch.float32": flow.float32,
        "torch.double": flow.double,
        "torch.int8": flow.int8,
        "torch.int32": flow.int32,
        "torch.int64": flow.int64,
        "torch.uint8": flow.uint8,
    }[str(mod)]


@torch2of.register
def _(mod: list, verbose=False) -> list:
    return [torch2of(m, verbose) for m in mod]


@torch2of.register
def _(mod: tuple, verbose=False) -> tuple:
    return tuple(torch2of(m, verbose) for m in mod)


@torch2of.register
def _(mod: OrderedDict, verbose=False) -> dict:
    return default_converter(mod, verbose, proxy_cls=OrderedDict)


@torch2of.register
def _(mod: set, verbose=False) -> set:
    return set(torch2of(m, verbose) for m in mod)


@torch2of.register(int)
@torch2of.register(float)
@torch2of.register(str)
@torch2of.register(bool)
def _(mod, verbose=False) -> Union[int, float, str, bool]:
    return mod


@torch2of.register
def _(mod: None, verbose=False) -> None:
    return mod


# TODO
@torch2of.register
def _(mod: flow.Tensor, verbose=False) -> None:
    return mod


@torch2of.register
def _(mod: types.BuiltinFunctionType, verbose=False) -> None:
    print(f"try to get function type mod {mod}")
    if hasattr(mod, "__module__"):
        mod_name = None
        if mod.__module__.startswith("torch._C._nn"):
            mod_name = mod.__module__.replace(
                "torch._C._nn", "oneflow._oneflow_internal._C"
            )
        elif mod.__module__.startswith("torch"):
            try:
                if getattr(torch.nn.functional, mod.__name__) == mod:
                    mod_name = "oneflow.nn.functional"
            except:
                mod_name = mod.__module__.replace("torch", "oneflow")
        if mod_name is not None:
            m = importlib.import_module(mod_name)
            return getattr(m, mod.__name__)

    return default_converter(mod, *args, **kwargs)
