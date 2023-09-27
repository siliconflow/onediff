print(f"\033[{32}m loading custom_interface_register.py \033[0m")
from .register import torch2of, replace_class

def _object_converter(obj, verbose=False):
    # ObjectConverter   obj -> of_obj
    # find proxy class
    new_obj_cls = replace_class(type(obj))

    def init(self):
        for k, v in obj.__dict__.items():
            attr = getattr(obj, k)
            self.__dict__[k] = torch2of(attr)

    of_obj_cls = type(str(new_obj_cls), (new_obj_cls,), {"__init__": init})
    of_obj = of_obj_cls()

    if verbose:
        print(f"convert {type(obj)} to {type(of_obj)}")
    return of_obj

try:
    import diffusers
    from .register import torch2of 

    @torch2of.register
    def _(mod:diffusers.models.attention_processor.AttnProcessor2_0, verbose=False):
        return _object_converter(mod, verbose=verbose)

    @torch2of.register
    def _(mod:diffusers.configuration_utils.FrozenDict, verbose=False):
        return _object_converter(mod, verbose=verbose)

except ImportError as e:
    print(f"\033[{31}m Waring: Failed to import {e=} \033[0m")