# ONEDIFF_TORCH_TO_ONEF_CLASS_MAP = { PYTORCH_MODEL_CLASS: ONEFLOW_MODEL_CLASS }
# ONEDIFF_CUSTOM_TORCH2OF_FUNC_TYPE_MAP = { Function :  TYPE }
import oneflow as flow
import diffusers_quant

ONEDIFF_TORCH_TO_ONEF_CLASS_MAP = {
    diffusers_quant.FakeQuantModule: diffusers_quant.OneFlowFakeQuantModule,
    diffusers_quant.StaticQuantConvModule: diffusers_quant.OneFlowStaticQuantConvModule,
    diffusers_quant.DynamicQuantConvModule: diffusers_quant.OneFlowDynamicQuantConvModule,
    diffusers_quant.StaticQuantLinearModule: diffusers_quant.OneFlowStaticQuantLinearModule,
    diffusers_quant.DynamicQuantLinearModule: diffusers_quant.OneFlowDynamicLinearQuantModule,
    diffusers_quant.models.attention_processor.TrtAttnProcessor: diffusers_quant.models.attention_processor_oneflow.OneFlowTrtAttnProcessor,
}


def convert_func(mod: flow.Tensor, verbose=False):
    return mod

ONEDIFF_CUSTOM_TORCH2OF_FUNC_TYPE_MAP = {
    convert_func: flow.Tensor,
}
