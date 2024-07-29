"""
Install:
    pip install pytest
Uasge:
    python -m pytest test_torch2of_demo.py
"""
import torch
import oneflow as flow  # usort: skip
import unittest

import numpy as np
from onediff.infer_compiler import oneflow_compile
from onediff.infer_compiler.backends.oneflow.transform import transform_mgr


class PyTorchModel(torch.nn.Module):
    """used torch2of conversion.

    For PyTorch models, input model must inherit torch.nn.Module to utilize trace and conversion of layers.
    """

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)

    def apply_model(self, x):
        return self.forward(x)


class OneFlowModel(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = flow.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)

    def apply_model(self, x):
        return self.forward(x)


class TestTorch2ofDemo(unittest.TestCase):
    def judge_tensor_func(self, y_pt, y_of):
        assert type(y_pt) == type(y_of)
        assert y_pt.device == y_of.device
        y_pt = y_pt.cpu().detach().numpy()
        y_of = y_of.cpu().detach().numpy()
        assert np.allclose(y_pt, y_of, atol=1e-3, rtol=1e-3)

    def test_torch2of_demo(self):
        # Register PyTorch model to OneDiff
        cls_key = transform_mgr.get_transformed_entity_name(PyTorchModel)
        transform_mgr.update_class_proxies({cls_key: OneFlowModel})

        # Compile PyTorch model to OneFlow
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pytorch_model = PyTorchModel().to(device)
        of_model = oneflow_compile(pytorch_model)

        # Verify conversion
        x = torch.randn(4, 4).to(device)

        #### 1. Use apply_model method
        y_pt = pytorch_model.apply_model(x)
        y_of = of_model.apply_model(x)
        self.judge_tensor_func(y_pt, y_of)

        #### 2. Use __call__ method
        y_pt = pytorch_model(x)
        y_of = of_model(x)
        self.judge_tensor_func(y_pt, y_of)


if __name__ == "__main__":
    unittest.main()
