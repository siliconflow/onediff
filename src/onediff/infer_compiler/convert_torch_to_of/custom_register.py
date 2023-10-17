import oneflow as flow
from ..import_tools import (
    print_red,
    print_green,
    get_mock_cls_name,
)
from .register import torch2of, default_converter
from ._globals import update_class_proxies, _initial_package_names


def import_diffusers():
    try:
        import diffusers
        from .mock_diffusers import Attention as mock_Attention
        from .mock_diffusers import AttnProcessor as mock_AttnProcessor
        from diffusers.models.attention_processor import Attention

        cls_key_value = {
            get_mock_cls_name(Attention): mock_Attention,
            get_mock_cls_name(
                diffusers.models.attention_processor.AttnProcessor2_0
            ): mock_AttnProcessor,
        }

        update_class_proxies(cls_key_value)

        @torch2of.register
        def _(
            mod: diffusers.models.attention_processor.AttnProcessor2_0, verbose=False
        ):
            return default_converter(mod, verbose=verbose)

    except ImportError as e:
        print_red(f"Failed to import diffusers {e=}")
        raise e


def import_comfy():
    try:
        import comfy
        from .mock_comfy import CrossAttentionPytorch, Linear

        cls_key = get_mock_cls_name(comfy.ldm.modules.attention.CrossAttentionPytorch)
        update_class_proxies({cls_key: CrossAttentionPytorch})

        cls_key = get_mock_cls_name(comfy.ops.Linear)
        update_class_proxies({cls_key: Linear})

        @torch2of.register
        def _(mod: comfy.latent_formats.SDXL, verbose=False):
            return default_converter(mod, verbose=verbose)

    except Exception as e:
        if "comfy" not in _initial_package_names:
            print_red(
                "Skipping import comfy,"
                "comfy not found in initial package names. "
                "Please export ONEDIFF_INITIAL_PACKAGE_NAMES_FOR_CLASS_PROXIES=diffusers,comfy, "
                "where 'diffusers' and 'comfy' are package names separated by commas."
            )
        else:
            print_red(f"Failed  {e=}")
            raise e


def import_diffusers_quant():
    try:

        import diffusers_quant

        cls_key_value = {
            get_mock_cls_name(
                diffusers_quant.FakeQuantModule
            ): diffusers_quant.OneFlowFakeQuantModule,
            get_mock_cls_name(
                diffusers_quant.StaticQuantConvModule
            ): diffusers_quant.OneFlowStaticQuantConvModule,
            get_mock_cls_name(
                diffusers_quant.DynamicQuantConvModule
            ): diffusers_quant.OneFlowDynamicQuantConvModule,
            get_mock_cls_name(
                diffusers_quant.StaticQuantLinearModule
            ): diffusers_quant.OneFlowStaticQuantLinearModule,
            get_mock_cls_name(
                diffusers_quant.DynamicQuantLinearModule
            ): diffusers_quant.OneFlowDynamicLinearQuantModule,
            get_mock_cls_name(
                diffusers_quant.models.attention_processor.TrtAttnProcessor
            ): diffusers_quant.models.attention_processor_oneflow.OneFlowTrtAttnProcessor,
        }
        update_class_proxies(cls_key_value)

        @torch2of.register
        def _(mod: flow.Tensor, verbose=False):
            return mod

    except ImportError as e:
        print_red(f"waring: Failed to import diffusers_quant {e=}")


print_green("Importing custom_register.py")
import_diffusers()
import_comfy()
import_diffusers_quant()
