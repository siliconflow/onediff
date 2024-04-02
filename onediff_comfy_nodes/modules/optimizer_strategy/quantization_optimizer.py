import os
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Any, Dict

import torch
import torch.nn as nn
from comfy.controlnet import ControlNet
from comfy.model_patcher import ModelPatcher
from onediff_quant.quantization import QuantizationConfig
from onediff_quant.quantization.module_operations import get_sub_module
from onediff_quant.quantization.quantize_calibrators import \
    QuantizationMetricsCalculator
from onediff_quant.quantization.quantize_config import Metric

from onediff.infer_compiler import oneflow_compile
from onediff.infer_compiler.with_oneflow_compile import DeployableModule

from ...utils.graph_path import generate_graph_path
from .optimizer_strategy import OptimizerStrategy, set_compiled_options


def get_torch_model(diff_model):
    if isinstance(diff_model, DeployableModule):
        return diff_model._torch_module
    else:
        return diff_model


class SubQuantizationPercentileCalculator(QuantizationMetricsCalculator):
    def __init__(
        self,
        model: torch.nn.Module,
        config: QuantizationConfig,
        cache_key: str = "",
        module_selector: callable = lambda x: x,
        *,
        seed=1,
        select=Metric.MAE.value,
        conv_percentage=0.9,
        linear_percentage=0.9,
    ):
        super().__init__(model, config, cache_key, module_selector, seed=seed)
        self.select = select
        self.conv_percentage = conv_percentage
        self.linear_percentage = linear_percentage

    @torch.no_grad()
    def calibrate(self, *args: Any, **kwargs: Any) -> Dict[str, Dict[str, float]]:
        if self.conv_percentage == 1.0 and self.linear_percentage == 1.0:
            # only_use_compute_density
            costs_calibrate_info = self.compute_quantization_costs(
                args, kwargs, module_selector=self.module_selector
            )
            costs_calibrate_info = self.apply_filter(costs_calibrate_info)
            self.save_quantization_status(
                costs_calibrate_info, "quantization_stats.json"
            )
            return costs_calibrate_info

        calibrate_info = self.calibrate_all_layers(
            args, kwargs, module_selector=self.module_selector
        )

        selected_model = self.module_selector(self.model)

        # Initialize max and min values, as well as lists for linear and convolutional layer data
        max_value, min_value = 2, -1
        linear_values, conv_values = [max_value, min_value], [max_value, min_value]
        # Iterate through quantization information for each layer, extracting quantization values for linear and convolutional layers
        for module_name, value_info in calibrate_info.items():
            module = get_sub_module(selected_model, module_name)
            values_list = (
                linear_values if isinstance(module, nn.Linear) else conv_values
            )
            values_list.append(float(value_info[self.select]))

        # Sort quantization values for linear and convolutional layers based on the selected evaluation metric
        linear_values.sort()
        conv_values.sort()

        # Calculate linear and convolutional thresholds
        conv_threshold = conv_values[int((len(conv_values) - 1) * self.conv_percentage)]
        linear_threshold = linear_values[
            int((len(linear_values) - 1) * self.linear_percentage)
        ]
        # print(f"Conv threshold: {conv_threshold}, Linear threshold: {linear_threshold}")

        if nn.Conv2d in self.config.module_settings:
            self.config.module_settings[nn.Conv2d][self.select] = conv_threshold

        if nn.Linear in self.config.module_settings:
            self.config.module_settings[nn.Linear][self.select] = linear_threshold

        # Apply filters and save quantization status information
        calibrate_info = self.apply_filter(calibrate_info)
        self.save_quantization_status(calibrate_info, "quantization_stats.json")

        return calibrate_info


@dataclass
class OnelineQuantizationOptimizerExecutor(OptimizerStrategy):
    conv_percentage: int = 60
    linear_percentage: int = 70
    conv_compute_density_threshold: int = 100
    linear_compute_density_threshold: int = 300

    @singledispatchmethod
    def apply(self, model, *args, **kwargs):
        print(f"{type(self).__name__}.apply() not implemented for {type(model)}")
        return model

    @apply.register(ModelPatcher)
    def _(self, model: ModelPatcher, ckpt_name=""):
        quant_config = QuantizationConfig.from_settings(
            quantize_conv=True,
            quantize_linear=True,
            bits=8,
            conv_mae_threshold=0.9,
            linear_mae_threshold=0.9,
            plot_calibrate_info=True,
            conv_compute_density_threshold=self.conv_compute_density_threshold,
            linear_compute_density_threshold=self.linear_compute_density_threshold,
        )
        diff_model = model.model.diffusion_model
        torch_model = get_torch_model(diff_model)
        quant_config.quantization_calculator = SubQuantizationPercentileCalculator(
            torch_model,
            quant_config,
            cache_key="unet",
            conv_percentage=self.conv_percentage / 100,
            linear_percentage=self.linear_percentage / 100,
        )
        if not isinstance(diff_model, DeployableModule):
            diff_model = oneflow_compile(diff_model)
        diff_model.apply_online_quant(quant_config)
        model.model.diffusion_model = diff_model

        graph_file = generate_graph_path(ckpt_name, model.model)
        quant_config.cache_dir = os.path.dirname(graph_file)
        set_compiled_options(diff_model, graph_file)
        quant_config = diff_model._deployable_module_quant_config
        return model

    @apply.register(ControlNet)
    def _(self, model, ckpt_name=""):
        quant_config = QuantizationConfig.from_settings(
            quantize_conv=True,
            quantize_linear=True,
            bits=8,
            conv_mae_threshold=0.9,
            linear_mae_threshold=0.9,
            plot_calibrate_info=True,
            conv_compute_density_threshold=self.conv_compute_density_threshold,
            linear_compute_density_threshold=self.linear_compute_density_threshold,
        )
        control_model = model.control_model
        quant_config.quantization_calculator = SubQuantizationPercentileCalculator(
            control_model,
            quant_config,
            cache_key="ControlNet",
            conv_percentage=self.conv_percentage / 100,
            linear_percentage=self.linear_percentage / 100,
        )
        graph_file = generate_graph_path(ckpt_name, control_model)
        quant_config.cache_dir = os.path.dirname(graph_file)
        if not isinstance(control_model, DeployableModule):
            control_model = oneflow_compile(control_model)
        control_model.apply_online_quant(quant_config)
        set_compiled_options(control_model, graph_file)
        model.control_model = control_model
        return model
