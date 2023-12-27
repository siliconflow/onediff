import numpy as np
from onediff.infer_compiler import oneflow_compile
from onediff.infer_compiler.transform import register
import torch
import torch.nn as nn
import oneflow as flow


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

def main():
    register(torch2oflow_class_map={MyModule: MyModuleOneflow})

    torch_module = MyModule().to("cuda")
    x = torch.randn(2, 10).to("cuda")
    y_torch = torch_module(x)

    oneflow_module = oneflow_compile(torch_module)
    y_oneflow = oneflow_module(x)

    for name, of_submodule in oneflow_module.named_modules():
        torch_submodule = torch_module.get_submodule(name)
        for param_name, oneflow_param in of_submodule.named_parameters():
            torch_param = torch_submodule.get_parameter(param_name)
            assert np.allclose(
                torch_param.data.detach().cpu().numpy(),
                oneflow_param.detach().cpu().numpy()
            )

if __name__ == "__main__":
    main()