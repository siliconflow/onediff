print(f"\033[{32}m loading custom_interface_register.py \033[0m")
from .register import torch2of, default_converter

try:
    import diffusers

    @torch2of.register
    def _(mod:diffusers.models.attention_processor.AttnProcessor2_0, verbose=False):
        return default_converter(mod, verbose=verbose)

    @torch2of.register
    def _(mod:diffusers.configuration_utils.FrozenDict, verbose=False):
        return default_converter(mod, verbose=verbose)

except ImportError as e:
    print(f"\033[{31}m Waring: Failed to import {e=} \033[0m")



try:
    import comfy 

    # <class 'comfy.latent_formats.SDXL'>
    @torch2of.register
    def _(mod:comfy.latent_formats.SDXL, verbose=False):
        return default_converter(mod, verbose=verbose)
    
except ImportError as e:
    print(f"\033[{31}m Waring: Failed to import {e=} \033[0m")