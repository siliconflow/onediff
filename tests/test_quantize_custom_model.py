import importlib
import os
import unittest

import oneflow as flow
import torch
from torch import nn

from onediff.infer_compiler import oneflow_compile
from onediff.infer_compiler.backends.oneflow.transform import register
from onediff.infer_compiler.backends.oneflow.utils.version_util import is_community_version

is_community = is_community_version()
onediff_quant_spec = importlib.util.find_spec("onediff_quant")
if is_community or onediff_quant_spec is None:
    print(f"{is_community=} {onediff_quant_spec=}")
    exit(0)

from onediff_quant.quantization import (
    OfflineQuantModule,
    OnlineQuantModule,
    QuantizationConfig,
    create_quantization_calculator,
)


# Define the model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Two convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        # Fully connected layer
        self.fc = nn.Linear(
            32 * 32 * 32, 4
        )  # Input channels are 32*32*32, output 4 classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor into one dimension
        x = self.fc(x)
        return x


class SimpleModel_OF(flow.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Two convolutional layers
        self.conv1 = flow.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = flow.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        # Fully connected layer
        self.fc = flow.nn.Linear(
            32 * 32 * 32, 4
        )  # Input channels are 32*32*32, output 10 classes

    def forward(self, x):
        x = flow.relu(self.conv1(x))
        x = flow.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor into one dimension
        x = self.fc(x)
        return x


register(torch2oflow_class_map={SimpleModel: SimpleModel_OF})

# Configure quantization
config = QuantizationConfig.from_settings(
    quantize_conv=True,
    quantize_linear=True,
    cache_dir="cache_dir",
    plot_calibrate_info=False,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleModel().to(device).to(torch.float16)
input_data = torch.randn(1, 3, 32, 32, dtype=torch.float16).to(
    device
)  # Input data size is [batch_size, channels, height, width]
seed = 1
calculator = create_quantization_calculator(model, config)
torch.manual_seed(seed)
standard_output = model(input_data)


class TestOnlineQuantModule(unittest.TestCase):
    def setUp(self):
        self.module = OnlineQuantModule(calculator, inplace=False)

    def test_quantize_with_calibration(self):
        quantized_model, info = self.module.quantize_with_calibration(input_data)
        status = self.module.collect_quantization_status(model, info)
        assert (
            status["quantized_conv_count"] == 2
            and status["quantized_linear_count"] == 1
        )
        compiled_model = oneflow_compile(quantized_model)
        torch.manual_seed(seed)
        quantized_output = compiled_model(input_data)
        # print(f'{quantized_output=} \n{standard_output=}')
        self.assertTrue(torch.allclose(standard_output, quantized_output, 1e4, 1e4))


class TestOfflineQuantModule(unittest.TestCase):
    def setUp(self):
        self.module = OfflineQuantModule(calculator, inplace=False)

    def test_quantize_with_calibration(self):
        quantized_model = self.module.quantize_with_calibration(input_data)[0]
        file_path = os.path.join(config.cache_dir, "quantized_model.pt")
        self.module.save(quantized_model, file_path)

        quantized_model = OfflineQuantModule(None).load(file_path=file_path)
        compiled_model = oneflow_compile(quantized_model)

        torch.manual_seed(seed)
        quantized_output = compiled_model(input_data)
        self.assertTrue(torch.allclose(standard_output, quantized_output, 1e4, 1e4))


if __name__ == "__main__":
    unittest.main()
