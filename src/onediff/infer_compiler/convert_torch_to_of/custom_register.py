from onediff.infer_compiler.import_tools import print_red, print_green
from .register import torch2of, default_converter
from ._globals import add_to_proxy_of_mds
from onediff.infer_compiler.import_tools import get_mock_cls_name
print_green("Importing custom_register.py")
try:
    import diffusers

    @torch2of.register
    def _(mod: diffusers.models.attention_processor.AttnProcessor2_0, verbose=False):
        return default_converter(mod, verbose=verbose)

    @torch2of.register
    def _(mod: diffusers.configuration_utils.FrozenDict, verbose=False):
        return default_converter(mod, verbose=verbose)
    
    from .mock_diffusers import Attention
    cls_key = get_mock_cls_name(diffusers.models.attention_processor.Attention)
    add_to_proxy_of_mds({cls_key:Attention})
  

except ImportError as e:
    print_red(f"Warning: Failed to import {e=}")


try:
    import comfy

    # <class 'comfy.latent_formats.SDXL'>
    @torch2of.register
    def _(mod: comfy.latent_formats.SDXL, verbose=False):
        return default_converter(mod, verbose=verbose)


except ImportError as e:
    print(f"\033[{31}m Warning: Failed to import {e=} \033[0m")
