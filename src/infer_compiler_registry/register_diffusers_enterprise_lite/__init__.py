from onediff.infer_compiler.backends.oneflow.transform import register

import oneflow as flow  # usort: skip
import diffusers_enterprise_lite

torch2oflow_class_map = {
    diffusers_enterprise_lite.ProxyFakeModule: diffusers_enterprise_lite.OneFlowFakeHPCModule,
    diffusers_enterprise_lite.ProxyDynamicConvModule: diffusers_enterprise_lite.OneFlowDynamicHPCConvModule,
    diffusers_enterprise_lite.ProxyDynamicLinearModule: diffusers_enterprise_lite.OneFlowDynamicHPCLinearModule,
}


def convert_func(mod: flow.Tensor, verbose=False):
    return mod


register(torch2oflow_class_map=torch2oflow_class_map, torch2oflow_funcs=[convert_func])
