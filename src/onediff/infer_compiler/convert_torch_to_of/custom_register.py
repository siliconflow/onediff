from onediff.infer_compiler.import_tools import print_red, print_green
from .register import torch2of, default_converter
print_green("Importing custom_register.py")
try:
    import diffusers

    @torch2of.register
    def _(mod: diffusers.models.attention_processor.AttnProcessor2_0, verbose=False):
        return default_converter(mod, verbose=verbose)

    @torch2of.register
    def _(mod: diffusers.configuration_utils.FrozenDict, verbose=False):
        return default_converter(mod, verbose=verbose)


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
