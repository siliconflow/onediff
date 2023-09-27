# register.py 
import torch
import oneflow as flow
from collections import OrderedDict
from functools import singledispatch
from .proxy import ProxySubmodule, replace_class


@singledispatch  
def torch2of(mod):
    raise NotImplementedError(f"Unsupported module type: {type(mod)}")

@torch2of.register
def _(mod: torch.nn.Module):
    proxy_md = ProxySubmodule(mod)
    
    # import pdb; pdb.set_trace()
    new_md_cls = replace_class(type(mod))
    
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
                self.__dict__[k] = attr

    def proxy_getattr(self, attr):
        nonlocal proxy_md

        if attr in ["_parameters", "_buffers", "_modules"]:
            raise ValueError(f"missing attr {attr} in base class")
        else:
            return getattr(proxy_md, attr)
    
    of_mod_cls = type(
        str(new_md_cls),
         (new_md_cls,), 
         {
        "__init__": init,
        "__getattr__": proxy_getattr
    })
    of_mod = of_mod_cls()
    return of_mod


@torch2of.register
def _(mod: torch.nn.ModuleList):
    of_mod_list = flow.nn.ModuleList()
    for original_submod in mod:
        submod = torch2of(original_submod)
        of_mod_list.append(submod)
    return of_mod_list

@torch2of.register
def _(mod: torch.nn.Sequential):
    
    of_mod_list = []
    for original_submod in mod:
        submod = torch2of(original_submod)
        of_mod_list.append(submod)
    return flow.nn.Sequential(*of_mod_list)
    

# ...其他注册    