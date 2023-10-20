import oneflow as torch
from ..proxy import proxy_class
import comfy

class Linear(proxy_class(comfy.ops.Linear)):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        import os
        if self.bias is not None and os.getenv("ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR") == "1":
            return torch._C.fused_matmul_bias(input, self.weight, self.bias)
        else:
            return torch.nn.functional.linear(input, self.weight, self.bias)