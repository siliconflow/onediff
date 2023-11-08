"""Convert torch object to oneflow object."""
from functools import singledispatch, partial
from collections import OrderedDict
from typing import Union

import importlib
import types
import torch
import oneflow as flow
from ..import_tools import print_red, print_yellow
from .proxy import ProxySubmodule, proxy_class
from ._globals import get_of_proxy_class

__all__ = ["torch2onef", "default_converter"]


@singledispatch
def torch2onef(mod, *args, **kwargs):
    return default_converter(mod, *args, **kwargs)


def default_converter(obj, verbose=False, *, proxy_cls=None):
    try:
        new_obj_cls = proxy_class(type(obj)) if proxy_cls is None else proxy_cls

        def init(self):
            for k, _ in obj.__dict__.items():
                attr = getattr(obj, k)
                self.__dict__[k] = torch2onef(attr)

        of_obj_cls = type(str(new_obj_cls), (new_obj_cls,), {"__init__": init})
        of_obj = of_obj_cls()

        if verbose:
            print(f"convert {type(obj)} to {type(of_obj)}")
        return of_obj
    except Exception as e:
        print_yellow(f"Unsupported type: {type(obj)}")
        # RuntimeError(f"Unsupported type: {type(obj)}")
        return obj


@torch2onef.register
def _(mod: torch.nn.Module, verbose=False):
    proxy_md = ProxySubmodule(mod)

    new_md_cls = proxy_class(type(mod))

    def init(self):
        nonlocal proxy_md

        # call the super `__init__` may cause unnecessary memory allocation,
        # so we call the nn.Module `__init__` instead.

        # super(type(self), self).__init__()
        flow.nn.Module.__init__(self)

        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._modules = OrderedDict()
        for (n, p) in list(proxy_md.named_parameters("", False)):
            self._parameters[n] = flow.nn.Parameter(
                flow.utils.tensor.from_torch(p.data), requires_grad=p.requires_grad
            )
        for (n, b) in list(proxy_md.named_buffers("", False)):
            self._buffers[n] = flow.utils.tensor.from_torch(b.data)
        for (n, m) in proxy_md._modules.items():
            self._modules[n] = torch2onef(m)

        for k, _ in proxy_md.__dict__.items():
            if k not in self.__dict__:
                attr = getattr(proxy_md, k)
                try:
                    self.__dict__[k] = torch2onef(attr)

                except Exception as e:
                    print_red(f"convert {type(attr)} failed: {e}")
                    raise NotImplementedError(f"Unsupported type: {type(attr)}")

    def proxy_getattr(self, attr):
        nonlocal proxy_md

        try:
            return super().__getattribute__(attr)
        except:
            if attr in self._modules:
                return self._modules[attr]
            if attr in self._parameters:
                return self._parameters[attr]
            elif attr in self._buffers:
                return self._buffers[attr]
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
                f"""
            Warning: {type(of_mod)} is in training mode 
            and is turned into eval mode which is good for infrence optimation.
            """
            )

    if verbose:
        print(f"convert {type(mod)} to {type(of_mod)}")

    return of_mod


@torch2onef.register
def _(mod: torch.nn.ModuleList, verbose=False):
    of_mod_list = flow.nn.ModuleList()
    for original_submod in mod:
        submod = torch2onef(original_submod, verbose)
        of_mod_list.append(submod)

    return of_mod_list


@torch2onef.register
def _(mod: torch.nn.Sequential, verbose=False):

    of_mod_list = []
    for original_submod in mod:
        submod = torch2onef(original_submod, verbose)
        of_mod_list.append(submod)
    of_mod_seq = flow.nn.Sequential(*of_mod_list)

    return of_mod_seq


@torch2onef.register
def _(mod: torch.nn.parameter.Parameter, verbose=False) -> flow.nn.Parameter:
    data = flow.utils.tensor.from_torch(mod.data)
    return flow.nn.Parameter(data, requires_grad=mod.requires_grad)


@torch2onef.register
def _(mod: torch.Tensor, verbose=False) -> flow.Tensor:
    return flow.utils.tensor.from_torch(mod)


@torch2onef.register
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


@torch2onef.register
def _(mod: list, verbose=False) -> list:
    return [torch2onef(m, verbose) for m in mod]


@torch2onef.register
def _(mod: tuple, verbose=False) -> tuple:
    return tuple(torch2onef(m, verbose) for m in mod)


@torch2onef.register
def _(mod: OrderedDict, verbose=False) -> dict:
    return default_converter(mod, verbose, proxy_cls=OrderedDict)


@torch2onef.register
def _(mod: set, verbose=False) -> set:
    return set(torch2onef(m, verbose) for m in mod)


@torch2onef.register(int)
@torch2onef.register(float)
@torch2onef.register(str)
@torch2onef.register(bool)
def _(mod, verbose=False) -> Union[int, float, str, bool]:
    return mod


@torch2onef.register
def _(mod: None, verbose=False) -> None:
    return mod


@torch2onef.register
def _(mod: types.BuiltinFunctionType, verbose=False) -> None:

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

    return default_converter(mod, verbose)


@torch2onef.register
def _(mod: torch.device, verbose=False):
    index = mod.index if mod.index is not None else 0
    return flow.device(mod.type, index)


@torch2onef.register
def _(mod: dict, verbose=False) -> dict:
    return {torch2onef(k): torch2onef(v, verbose) for k, v in mod.items()}


@torch2onef.register
def _(func: types.FunctionType, verbose=False):
    proxy_obj = get_of_proxy_class(func)
    new_func = getattr(proxy_obj, func.__name__)
    return new_func


@torch2onef.register
def _(mod: partial, verbose=False):
    # https://docs.python.org/3/library/functools.html?highlight=partial#functools.partial
    func = torch2onef(mod.func)
    args = torch2onef(mod.args)
    keywords = torch2onef(mod.keywords)
    return partial(func, *args, **keywords)
