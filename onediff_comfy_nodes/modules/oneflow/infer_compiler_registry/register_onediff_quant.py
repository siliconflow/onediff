import onediff_quant
import oneflow as flow  # usort: skip
from onediff.infer_compiler.backends.oneflow.transform import register

torch2oflow_class_map = {
    onediff_quant.FakeQuantModule: onediff_quant.OneFlowFakeQuantModule,
    onediff_quant.StaticQuantConvModule: onediff_quant.OneFlowStaticQuantConvModule,
    onediff_quant.DynamicQuantConvModule: onediff_quant.OneFlowDynamicQuantConvModule,
    onediff_quant.StaticQuantLinearModule: onediff_quant.OneFlowStaticQuantLinearModule,
    onediff_quant.DynamicQuantLinearModule: onediff_quant.OneFlowDynamicLinearQuantModule,
}


def convert_func(mod: flow.Tensor, verbose=False):
    return mod


register(torch2oflow_class_map=torch2oflow_class_map, torch2oflow_funcs=[convert_func])
