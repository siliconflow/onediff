import numpy as np
import torch
import torch.nn as nn
from onediff.infer_compiler import oneflow_compile
from onediff.infer_compiler.backends.oneflow.transform import register
import oneflow as flow  # usort: skip


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x


class MyModuleOneflow(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = flow.nn.ModuleList([flow.nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x


register(torch2oflow_class_map={MyModule: MyModuleOneflow})

m = MyModule().to("cuda")
x = torch.randn(2, 10).to("cuda")
y_torch = m(x)

m = oneflow_compile(m)
y_oneflow = m(x)

assert np.allclose(y_torch.detach().cpu(), y_oneflow.detach().cpu(), 1e-03, 1e-03)

from onediff.infer_compiler.backends.oneflow.dual_module import (
    DualModule,
    DualModuleList,
)

assert isinstance(m.linears, DualModuleList)

x = getattr(m.linears, "1")
assert isinstance(x, DualModule)

x.bias = None
setattr(m.linears, "2", x)

assert m.linears[2].bias is None
assert m.linears._torch_modules[2].bias is None
assert m.linears._oneflow_modules[2].bias is None

m.linears[3] = x

assert m.linears[3].bias is None
assert m.linears._torch_modules[3].bias is None
assert m.linears._oneflow_modules[3].bias is None
