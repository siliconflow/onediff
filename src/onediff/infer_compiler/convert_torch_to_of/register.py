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
import torch
import oneflow as flow
from typing import Union
from collections import OrderedDict
from functools import singledispatch
from .proxy import ProxySubmodule, proxy_class

__all__ = ["torch2of", "default_converter"]


@singledispatch
def torch2of(mod, *args, **kwargs):
    try:

        return default_converter(mod, *args, **kwargs)
    except Exception as e:
        print(f"convert {type(mod)} failed: {e}")
        raise NotImplementedError(f"Unsupported type: {type(mod)}")


@torch.no_grad()
def default_converter(obj, verbose=False, *, proxy_cls=None):
    # ObjectConverter   obj -> of_obj find proxy class
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
        raise NotImplementedError(f"Unsupported type: {type(obj)}")


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
def _(mod: dict, verbose=False) -> dict:
    return {k: torch2of(v, verbose) for k, v in mod.items()}


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
