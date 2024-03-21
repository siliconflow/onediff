import json
import os
from typing import Any, Dict, Optional
import torch.nn as nn
from dataclasses import dataclass, field
from enum import Enum


class Metric(Enum):
    MAE = "mae"
    MSE = "mse"
    SSIM = "ssim"
    COMPUTE_DENSITY = "compute_density"


@dataclass
class QuantizationConfig:
    bits: int = 8
    inplace: bool = True
    module_settings: Dict[type, Dict[str, float]] = field(default_factory=lambda: {
        nn.Conv2d: {
            Metric.MAE.value: 1.0,
            Metric.MSE.value: 1.0,
            Metric.SSIM.value: 0.0,
            Metric.COMPUTE_DENSITY.value: 0.0,
        },
        nn.Linear: {
            Metric.MAE.value: 1.0,
            Metric.MSE.value: 1.0,
            Metric.SSIM.value: 0.0,
            Metric.COMPUTE_DENSITY.value: 0.0,
        },
    })
    auto_quantization_level: int = None
    plot_calibrate_info: bool = False
    calibrate_info: Dict[str, Dict[str, Any]] = None
    cache_dir: str = None

    @classmethod
    def from_dict(cls, config: Dict):
        return cls(**config)

    def save_calibrate_info(self, file_name: str, calibrate_info: Dict):
        if self.cache_dir is None:
            return

        if os.path.exists(self.cache_dir) is False:
            os.makedirs(self.cache_dir)

        file_path = os.path.join(self.cache_dir, file_name)
        json.dump(calibrate_info, open(file_path, "w"))

    def load_calibrate_info(self, file_name: str) -> Dict:
        if self.cache_dir is None:
            return {}

        file_path = os.path.join(self.cache_dir, file_name)
        if not os.path.exists(file_path):
            return {}

        json_data = json.load(open(file_path, "r"))
        return json_data

    def is_valid_layer(
        self, layer, additional_info: Optional[Dict[str, float]] = None
    ) -> bool:
        """Determines if a layer is valid for quantization based on provided criteria.

        Args:
            layer: The layer to be validated.
            additional_info: Additional information needed for validation (e.g., SSIM, MAE, MSE).

        Returns:
            bool: True if the layer is valid for quantization, False otherwise.
        """
        # Look for the type of the layer in module settings
        layer_type = None
        for quantize_type in self.module_settings.keys():
            if isinstance(layer, quantize_type):
                layer_type = quantize_type
                break

        if layer_type is None:
            # If layer type is not found, it's not valid for quantization
            return False

        if additional_info is not None:
            # Retrieve threshold values from module settings
            thresholds = self.module_settings[layer_type]
            # Define conditions for each metric
            conditions = {
                Metric.MAE.value: (lambda x: x > thresholds.get(Metric.MAE.value, 1.0)),
                Metric.MSE.value: (lambda x: x > thresholds.get(Metric.MSE.value, 1.0)),
                Metric.SSIM.value: (
                    lambda x: x < thresholds.get(Metric.SSIM.value, 0.0)
                ),
                Metric.COMPUTE_DENSITY.value: (
                    lambda x: x < thresholds.get(Metric.COMPUTE_DENSITY.value, 0.0)
                ),
            }
            # Check if each metric is met
            if any(
                metric in additional_info and condition(additional_info[metric])
                for metric, condition in conditions.items()
            ):
                return False

        # If all conditions are met, the layer is valid for quantization
        return True

    def save_quantization_config(self, file_name: str):
        if self.cache_dir is None:
            return

        if os.path.exists(self.cache_dir) is False:
            os.makedirs(self.cache_dir)

        file_path = os.path.join(self.cache_dir, file_name)
        json.dump(self.__dict__, open(file_path, "w"))
    

    @classmethod
    def from_settings(cls, quantize_conv=True, quantize_linear=True, bits=8, *,
                            inplace=True, calibrate_info:Dict[str,Any]=None, cache_dir:str=None, plot_calibrate_info=False,
                            conv_ssim_threshold=0.0, linear_ssim_threshold=0.0, conv_mae_threshold=1.0, linear_mae_threshold=1.0,
                            conv_mse_threshold=1.0, linear_mse_threshold=1.0, conv_compute_density_threshold=0.0,
                            linear_compute_density_threshold=0.0):
        
        module_settings = {}
        if quantize_conv:
            module_settings[nn.Conv2d] = {
                Metric.SSIM.value: conv_ssim_threshold,
                Metric.MAE.value: conv_mae_threshold,
                Metric.MSE.value: conv_mse_threshold,
                Metric.COMPUTE_DENSITY.value: conv_compute_density_threshold,
            }
        if quantize_linear:
            module_settings[nn.Linear] = {
                Metric.SSIM.value: linear_ssim_threshold,
                Metric.MAE.value: linear_mae_threshold,
                Metric.MSE.value: linear_mse_threshold,
                Metric.COMPUTE_DENSITY.value: linear_compute_density_threshold,
            }

        return cls(
            bits=bits,
            inplace=inplace,
            calibrate_info=calibrate_info,
            cache_dir=cache_dir,
            plot_calibrate_info=plot_calibrate_info,
            module_settings=module_settings
        )