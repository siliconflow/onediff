import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
from ..utils.log_utils import logger
from ..utils.graph_management_utils import calculate_model_hash
from ...optimization.quant_optimizer import quantize_sub_module


class Pipe:
    def __init__(
        self, model: torch.nn.Module, args: Tuple[Any], kwargs: Dict[str, Any]
    ):
        self.model = model
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwds):
        return self.model(*self.args, **self.kwargs)


def unwrap_one(obj: Union[Tuple[Any], Any]) -> Any:
    if isinstance(obj, (tuple, list)):
        assert len(obj) == 1
        return obj[0]
    else:
        return obj


def normalize(data):
    min_val, max_val = min(data), max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]


def find_peaks(lst):
    peaks = []
    # Check the first element
    if lst[0] > lst[1]:
        peaks.append(0)
    # Check the middle elements
    for i in range(1, len(lst) - 1):
        if lst[i] > lst[i - 1] and lst[i] > lst[i + 1]:
            peaks.append(i)
    # Check the last element
    if lst[-1] > lst[-2]:
        peaks.append(len(lst) - 1)
    return peaks


@dataclass
class QuantizationConfig:
    """QuantizationConfig
    ### Introduction:
        The `QuantizationConfig` class is designed for configuring the process of quantization.

    ### References
        - [onediff_quant/quantization/quantize_pipeline.py](https://github.com/siliconflow/onediff-quant/blob/544ab9c179fabc1953b0b9757275c3751ab3e304/onediff_quant/quantization/quantize_pipeline.py)
        - [tools/quantize-svd-fast.py](https://github.com/siliconflow/onediff-quant/blob/544ab9c179fabc1953b0b9757275c3751ab3e304/tools/quantize-svd-fast.py)
        - [Dev quantize pipeline](https://github.com/siliconflow/onediff/pull/704)
        - [ComfyUI model quantification tool](https://github.com/siliconflow/sd-team/issues/193#issuecomment-1893290387)
    
    ### Test Experimental Data
        - [00](https://github.com/siliconflow/onediff/pull/495#discussion_r1448999556)

    """

    bits: int = 8
    inplace: bool = True
    quantize_conv: bool = True
    quantize_linear: bool = True
    compute_density_threshold: int = 900
    conv_mse_threshold: float = 0.1
    linear_mse_threshold: float = 0.1
    calibrate_info: Dict[str, Any] = None
    use_quantization: bool = False
    cache_dir: str = "."

    def save_calibrate_info(self, file_name: str, calibrate_info: Dict[str, float]):
        file_path = os.path.join(self.cache_dir, file_name)
        json.dump(calibrate_info, open(file_path, "w"))

    def load_calibrate_info(self, file_name: str) -> Dict[str, float]:
        file_path = os.path.join(self.cache_dir, file_name)
        if not os.path.exists(file_path):
            return {}
        json_data = json.load(open(file_path, "r"))
        return json_data


class QuantizationModule:
    def __init__(
        self, torch_model: torch.nn.Module, quantization_config: QuantizationConfig
    ):
        self.torch_model = torch_model
        self.quantization_config = quantization_config

    def quantize_forward(self, *args: Any, **kwargs: Any) -> None:
        if not self.quantization_config.use_quantization:
            return
        config = self.quantization_config
        from onediff_quant.utils import find_quantizable_modules
        from onediff_quant.utils import metric_quantize_costs

        pipe = Pipe(self.torch_model, args, kwargs)
        basic_cache_key = (
            type(self.torch_model).__name__
            + "_"
            + calculate_model_hash(self.torch_model)
        )
        quantizable_modules: Dict[str, nn.Module] = find_quantizable_modules(
            pipe.model, module_cls=[nn.Linear, nn.Conv2d]
        )

        costs_file_name = basic_cache_key + "_costs_calibrate_info.json"
        costs_calibrate_info: Dict[str, Dict[str, float]] = config.load_calibrate_info(
            costs_file_name
        )
        if len(costs_calibrate_info) == 0:
            costs = metric_quantize_costs(pipe, {}, quantizable_modules)
            costs_calibrate_info = {
                name: {"compute_density": costs.get_compute_density(name)}
                for name in quantizable_modules.keys()
            }
            config.save_calibrate_info(
                calibrate_info=costs_calibrate_info, file_name=costs_file_name
            )
            logger.info("Save Costs Calibrate Info to %s", costs_file_name)
        else:
            logger.info("Load Costs Calibrate Info from %s", costs_file_name)

        # standard output
        standard_output = self.torch_model(*args, **kwargs)
        standard_output = unwrap_one(standard_output)

        # calibrate_linear
        linears_file_name = basic_cache_key + "_linears_calibrate_info.json"
        linears_calibrate_info: Dict[
            str, Dict[str, float]
        ] = config.load_calibrate_info(linears_file_name)
        if len(linears_calibrate_info) == 0 and config.quantize_linear:
            linear_modules = {
                key: v
                for key, v in quantizable_modules.items()
                if isinstance(v, nn.Linear)
            }
            for sub_name, layer in tqdm(
                linear_modules.items(), desc="Linear Quantization"
            ):
                handle = quantize_sub_module(
                    model=pipe.model, sub_name=sub_name, sub_module=layer
                )
                # Current Output
                current_output = pipe.model(*args, **kwargs)
                current_output = unwrap_one(current_output)
                mse = torch.mean((standard_output - current_output) ** 2)
                mae = torch.mean(torch.abs(standard_output - current_output))
                max_diff = torch.max(torch.abs(standard_output - current_output))
                linears_calibrate_info[sub_name] = {
                    "mse": mse.item(),
                    "mae": mae.item(),
                    "max_diff": max_diff.item(),
                }
                handle()
            logger.info("Save Linear Calibrate Info to %s", linears_file_name)
            config.save_calibrate_info(
                calibrate_info=linears_calibrate_info, file_name=linears_file_name
            )
        else:
            logger.info("Load Linear Calibrate Info from %s", linears_file_name)

        # calibrate_conv2d
        conv2d_file_name = basic_cache_key + "_conv2d_calibrate_info.json"
        conv2d_calibrate_info: Dict[str, Dict[str, float]] = config.load_calibrate_info(
            conv2d_file_name
        )
        if len(conv2d_calibrate_info) == 0 and config.quantize_conv:
            conv2d_moudules = {
                key: v
                for key, v in quantizable_modules.items()
                if isinstance(v, nn.Conv2d)
            }

            for sub_name, layer in tqdm(
                conv2d_moudules.items(), desc="Conv Quantization"
            ):
                handle = quantize_sub_module(
                    model=pipe.model, sub_name=sub_name, sub_module=layer
                )
                # Current Output
                current_output = pipe.model(*args, **kwargs)
                current_output = unwrap_one(current_output)
                mse = torch.mean((standard_output - current_output) ** 2)
                mae = torch.mean(torch.abs(standard_output - current_output))
                max_diff = torch.max(torch.abs(standard_output - current_output))
                conv2d_calibrate_info[sub_name] = {
                    "mse": mse.item(),
                    "mae": mae.item(),
                    "max_diff": max_diff.item(),
                }
                handle()
            logger.info("Save Conv2d Calibrate Info to %s", conv2d_file_name)
            config.save_calibrate_info(
                calibrate_info=conv2d_calibrate_info, file_name=conv2d_file_name
            )
        else:
            logger.info("Load Conv2d Calibrate Info from %s", conv2d_file_name)

        mae_lst, mse_lst, max_diff_lst = [], [], []
        linear_conv_names_lst = []
        for name, layer in quantizable_modules.items():
            if config.quantize_conv and isinstance(layer, nn.Conv2d):
                mae_lst.append(conv2d_calibrate_info[name]["mae"])
                mse_lst.append(conv2d_calibrate_info[name]["mse"])
                max_diff_lst.append(conv2d_calibrate_info[name]["max_diff"])
                linear_conv_names_lst.append(name)
            elif config.quantize_linear and isinstance(layer, nn.Linear):
                mae_lst.append(linears_calibrate_info[name]["mae"])
                mse_lst.append(linears_calibrate_info[name]["mse"])
                max_diff_lst.append(linears_calibrate_info[name]["max_diff"])
                linear_conv_names_lst.append(name)

        mae_lst, mse_lst, max_diff_lst = (
            normalize(mae_lst),
            normalize(mse_lst),
            normalize(max_diff_lst),
        )
        indexs = find_peaks(mse_lst) + find_peaks(mae_lst) + find_peaks(max_diff_lst)
        indexs = set(indexs)
        result_calibrate_info = {}
        for i, name in enumerate(linear_conv_names_lst):
            if i not in indexs:
                result_calibrate_info[name] = "1"
        config.calibrate_info = result_calibrate_info
