import importlib
import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Any, Dict, Type
from abc import ABC, abstractmethod
from functools import singledispatch
from .quantize_config import (
    QuantizationConfig,
    Metric,
)
from .quantize_utils import set_random_seed
from ..infer_compiler.utils.log_utils import logger
from ..infer_compiler.utils.module_operations import get_sub_module, quantize_sub_module
from ..infer_compiler.utils.version_util import is_community_version
from ..infer_compiler.utils.graph_management_utils import calculate_model_hash


__all__ = [
    "QuantizationCalibratorInterface",
    "OnlineQuantDeployableModule",
    "CostsQuantizationCalibrator",
    "QuantizationMetricsCalculator",
    "DiffusersCalibrator",
]


class QuantizationCalibratorInterface(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        config: QuantizationConfig,
        cache_key: str = "",
        module_selector: callable = lambda x: x,
        *,
        seed=1,
    ):
        self.model = model
        self.config = config
        self.cache_key = cache_key
        self.module_selector = module_selector
        self.seed = seed

    @abstractmethod
    def calibrate(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement calibrate method.")

    @abstractmethod
    def plot_calibrate_info(self, *args: Any, **kwargs: Any):
        pass


class OnlineQuantDeployableModule(ABC):
    def calibrate_forward(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement calibrate_forward method.")

    def is_quant_enabled(self) -> bool:
        raise NotImplementedError("Subclasses must implement is_quantizable method.")


@singledispatch
def calculate_output_metrics(standard_output, current_output) -> Dict:
    raise NotImplementedError(
        f"{type(standard_output)} is not supported for calculate_output_metrics"
    )


@calculate_output_metrics.register(torch.Tensor)
def _(standard_output, current_output):
    mse = torch.mean((standard_output - current_output) ** 2)
    mae = torch.mean(torch.abs(standard_output - current_output))

    return {
        Metric.MAE.value: mae.item(),
        Metric.MSE.value: mse.item(),
    }


@calculate_output_metrics.register(list)
@calculate_output_metrics.register(tuple)
def _(standard_output, current_output):
    if len(standard_output) != len(current_output):
        raise RuntimeError("Output length mismatch")

    if len(standard_output) == 1:
        return calculate_output_metrics(standard_output[0], current_output[0])
    else:
        raise NotImplementedError(
            "Subclasses must implement calculate_output_metrics method."
        )


class CostsQuantizationCalibrator(QuantizationCalibratorInterface):
    def __init__(
        self,
        model: torch.nn.Module,
        config: QuantizationConfig,
        cache_key: str = "",
        module_selector: callable = lambda x: x,
        *,
        seed=1,
    ):
        super().__init__(model, config, cache_key, module_selector, seed=seed)

    def calibrate(self, *args: Any, **kwargs: Any) -> Dict[str, Dict[str, float]]:
        selected_model = self.module_selector(self.model)
        costs_calibrate_info = self.compute_quantization_costs(
            args, kwargs, module_selector=self.module_selector
        )
        compute_density_key = Metric.COMPUTE_DENSITY.value
        filtered_costs_calibrate_info = {}

        for key, value in costs_calibrate_info.items():
            layer = get_sub_module(selected_model, key)
            if self.config.is_valid_layer(
                layer, {compute_density_key: value[compute_density_key]}
            ):
                filtered_costs_calibrate_info[key] = value

        if self.config.plot_calibrate_info and self.config.cache_dir is not None:
            plot_calibrate_info_file_name = f"{type(self).__name__}_{self.cache_key}"
            file_path = os.path.join(
                self.config.cache_dir, plot_calibrate_info_file_name
            )
            self.plot_calibrate_info(file_path, costs_calibrate_info)

        return filtered_costs_calibrate_info

    def plot_calibrate_info(
        self, file_path: str, calibrate_info: Dict[str, Dict[str, float]]
    ):
        import plotly.graph_objs as go
        from plotly.offline import plot

        fig = go.Figure()
        data = []
        colors = ["blue", "red", "green", "orange", "purple"]
        for metric, color in zip(Metric, colors):
            try:
                key = metric.value
                trace = go.Scatter(
                    x=list(range(len(calibrate_info))),
                    y=[value[key] for value in calibrate_info.values()],
                    mode="lines",
                    name=key,
                    line=dict(color=color),
                )
                data.append(trace)
            except KeyError:
                pass

        layout = go.Layout(
            title="Calibrate Info",
            xaxis=dict(title="index"),
            yaxis=dict(title="value"),
        )
        fig = go.Figure(data=data, layout=layout)
        plot(fig, filename=file_path)

    def compute_quantization_costs(
        self, args, kwargs, module_selector=lambda x: x
    ) -> Dict[str, Dict[str, float]]:
        from onediff_quant.utils import metric_quantize_costs, find_quantizable_modules

        config = self.config
        compute_density_key = Metric.COMPUTE_DENSITY.value
        selected_model = module_selector(self.model)
        quantizable_modules = find_quantizable_modules(
            selected_model, module_cls=list(config.module_settings.keys())
        )
        costs = metric_quantize_costs(self.model, args, kwargs, quantizable_modules)
        costs_calibrate_info = {
            name: {compute_density_key: costs.get_compute_density(name)}
            for name in quantizable_modules.keys()
        }
        return costs_calibrate_info


class QuantizationMetricsCalculator(CostsQuantizationCalibrator):
    def __init__(
        self,
        model: torch.nn.Module,
        config: QuantizationConfig,
        cache_key: str = "",
        module_selector: callable = lambda x: x,
        *,
        seed=1,
    ):
        super().__init__(model, config, cache_key, module_selector, seed=seed)

    def calibrate_all_layers(
        self, args, kwargs, module_selector=lambda x: x
    ) -> Dict[str, Dict[str, float]]:
        config = self.config

        set_random_seed(self.seed)
        standard_output = self.model(*args, **kwargs)
        selected_model = module_selector(self.model)

        calibrate_info = {}
        costs_calibrate_info = self.compute_quantization_costs(
            args, kwargs, module_selector
        )

        for quantize_type in config.module_settings.keys():
            quantizable_modules = {
                name: layer
                for name, layer in selected_model.named_modules()
                if isinstance(layer, quantize_type)
            }
            if not quantizable_modules:
                continue
            layer_name = quantize_type.__name__
            logger.info(f"Quantize {layer_name} Layers")
            file_name = f"{layer_name}_{self.cache_key}.json"
            info = self._calibrate_modules(
                file_name=file_name,
                modules=quantizable_modules,
                desc=f"{layer_name} Layers",
                standard_output=standard_output,
                args=args,
                kwargs=kwargs,
                module_selector=module_selector,
            )
            calibrate_info.update(info)

        # Normalize mae and mse values
        for select in [Metric.MAE.value, Metric.MSE.value]:
            min_val = min([x[select] for x in calibrate_info.values()])
            max_val = max([x[select] for x in calibrate_info.values()])
            for key, value in calibrate_info.items():
                value[select] = (value[select] - min_val) / (max_val - min_val)

        # Add compute density to calibrate_info
        density_key = Metric.COMPUTE_DENSITY.value
        for key, value in calibrate_info.items():
            compute_density = costs_calibrate_info[key][density_key]
            value[density_key] = compute_density

        if self.config.cache_dir is not None:
            file_name = f"calibrate_info_{self.cache_key}.html"
            file_path = os.path.join(self.config.cache_dir, file_name)
            self.plot_calibrate_info(file_path, calibrate_info)
        return calibrate_info

    def _calibrate_modules(
        self,
        file_name: str,
        modules: Dict[str, nn.Module],
        desc: str,
        standard_output: torch.tensor,
        args,
        kwargs,
        module_selector=lambda x: x,
    ) -> Dict[str, Dict[str, float]]:

        calibrate_info: Dict[str, Dict[str, float]] = self.config.load_calibrate_info(
            file_name=file_name
        )

        if len(calibrate_info) > 0:
            return calibrate_info

        selected_model = module_selector(self.model)
        for sub_name, layer in tqdm(modules.items(), desc=desc):
            logger.debug(f"Quantize {sub_name}")
            handle = quantize_sub_module(
                model=selected_model, sub_name=sub_name, sub_module=layer
            )

            set_random_seed(self.seed)
            current_output = self.model(*args, **kwargs)
            calibrate_info[sub_name] = calculate_output_metrics(
                standard_output, current_output
            )
            handle()

        self.config.save_calibrate_info(
            calibrate_info=calibrate_info, file_name=file_name
        )
        logger.info(f"Saved {desc} Calibrate Info to {file_name}")
        return calibrate_info

    @torch.no_grad()
    def calibrate(self, *args: Any, **kwargs: Any) -> Dict[str, Dict[str, float]]:
        calibrate_info = self.calibrate_all_layers(
            args, kwargs, module_selector=self.module_selector
        )
        selected_model = self.model
        return {
            key: value
            for key, value in calibrate_info.items()
            if self.config.is_valid_layer(get_sub_module(selected_model, key), value)
        }


class DiffusersCalibrator(QuantizationMetricsCalculator):
    def __init__(
        self,
        pipe,
        config: QuantizationConfig,
        cache_key: str,
        module_selector=lambda pipe: pipe.unet,
        *,
        seed=1,
    ):
        super().__init__(pipe, config, cache_key, module_selector, seed=seed)

    def calibrate(self, *args, **kwargs) -> Dict[str, Any]:
        calibrate_info = self.calibrate_all_layers(
            args, kwargs, module_selector=self.module_selector
        )

        selected_model = self.module_selector(self.model)

        return {
            key: value
            for key, value in calibrate_info.items()
            if self.config.is_valid_layer(get_sub_module(selected_model, key), value)
        }


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

    def calibrate(self, *args: Any, **kwargs: Any) -> Dict[str, Dict[str, float]]:
        calibrate_info = self.calibrate_all_layers(
            args, kwargs, module_selector=self.module_selector
        )
        selected_model = self.module_selector(self.model)

        # extract_data
        linear_datas, conv_datas = [], []
        for key, value in calibrate_info.items():
            module = get_sub_module(selected_model, key)
            if isinstance(module, nn.Linear):
                linear_datas.append(float(value[self.select]))
            elif isinstance(module, nn.Conv2d):
                conv_datas.append(float(value[self.select]))

        if self.select == Metric.SSIM.value:
            linear_datas.sort()
            conv_datas.sort()
        else:
            linear_datas.sort(reverse=True)
            conv_datas.sort(reverse=True)

        conv_th = conv_datas[int(len(conv_datas) * self.conv_percentage)]
        linear_th = linear_datas[int(len(linear_datas) * self.linear_percentage)]
        logger.info(f"Conv threshold: {conv_th}, Linear threshold: {linear_th}")

        if nn.Conv2d in self.config.module_settings:
            self.config.module_settings[nn.Conv2d][self.select] = conv_th

        if nn.Linear in self.config.module_settings:
            self.config.module_settings[nn.Linear][self.select] = linear_th

        return {
            key: value
            for key, value in calibrate_info.items()
            if self.config.is_valid_layer(get_sub_module(selected_model, key), value)
        }


def create_quantization_calculator(
    model: torch.nn.Module,
    config: QuantizationConfig,
    module_selector=lambda x: x,
    seed: int = 1,
    *,
    calibrator_type: Type[QuantizationCalibratorInterface] = None,
):

    basic_cache_key = type(model).__name__ + "_" + calculate_model_hash(model)
    if calibrator_type is not None:
        return calibrator_type(model, config, basic_cache_key, module_selector, seed)
    elif config.auto_quantization_level is None:
        return QuantizationMetricsCalculator(
            model, config, basic_cache_key, module_selector=module_selector, seed=seed,
        )
    else:
        raise NotImplementedError("auto_quantization_level is not supported")


class OnlineQuantModule(OnlineQuantDeployableModule):
    def __init__(self, torch_module: nn.Module):
        self.model = torch_module
        self.quantization_config: QuantizationConfig = None

    def calibrate_forward(self, *args: Any, **kwargs: Any):
        calculator = create_quantization_calculator(
            self.model, self.quantization_config
        )
        out = calculator.calibrate(*args, **kwargs)
        return out

    def is_quant_enabled(self):
        if is_community_version():
            return False

        if self.quantization_config is None:
            return False

        if self.quantization_config.calibrate_info is not None:
            return False

        return True


if importlib.util.find_spec("diffusers") is not None:
    from diffusers.utils import BaseOutput
    from skimage.metrics import structural_similarity

    @calculate_output_metrics.register(BaseOutput)
    def _(standard_output, current_output):
        def extract_images(output):
            if hasattr(output, "frames"):
                return output.frames[0][-1]
            elif hasattr(output, "images"):
                return output.images[0]
            else:
                raise ValueError("invalid results")

        img1 = extract_images(standard_output)
        img2 = extract_images(current_output)
        img1, img2 = np.asarray(img1), np.asarray(img2)
        ssim = structural_similarity(img1, img2, channel_axis=2)
        mse = ((img1 - img2) ** 2).mean()
        mae = abs(img1 - img2).mean()
        logger.info(f"SSIM: {ssim}, MSE: {mse}, MAE: {mae}")
        return {
            Metric.SSIM.value: ssim,
            Metric.MSE.value: mse,
            Metric.MAE.value: mae,
        }
