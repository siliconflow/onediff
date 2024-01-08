from onediff.infer_compiler import register

import oneflow as flow
import diffusers_quant

torch2oflow_class_map = {
    diffusers_quant.ProxyFakeModule: diffusers_quant.OneFlowFakeQuantModule,
    diffusers_quant.ProxyStaticConvModule: diffusers_quant.OneFlowStaticQuantConvModule,
    diffusers_quant.ProxyDynamicConvModule: diffusers_quant.OneFlowDynamicQuantConvModule,
    diffusers_quant.ProxyStaticLinearModule: diffusers_quant.OneFlowStaticQuantLinearModule,
    diffusers_quant.ProxyDynamicLinearModule: diffusers_quant.OneFlowDynamicLinearQuantModule,
}


def convert_func(mod: flow.Tensor, verbose=False):
    return mod


register(torch2oflow_class_map=torch2oflow_class_map, torch2oflow_funcs=[convert_func])
