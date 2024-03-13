import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union
from ..utils.module_operations import get_sub_module
from ..utils.log_utils import logger
from ..utils.graph_management_utils import calculate_model_hash
from ...optimization.quant_optimizer import quantize_sub_module


class QuantizationCalibratorInterface(ABC):
    @abstractmethod
    def calibrate(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        pass

    @abstractmethod
    def plot_calibrate_info(self, *args: Any, **kwargs: Any):
        pass


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
        - [mse and ssim](https://github.com/siliconflow/onediff/pull/495#discussion_r1448999556)

    """

    bits: int = 8
    inplace: bool = True
    quantize_conv: bool = True
    quantize_linear: bool = True
    quality_level: int = 6
    plot_calibrate_info: bool = False
    compute_density_threshold: int = 10
    calibrate_info: Dict[str, Any] = None
    use_quantization: bool = False
    cache_dir: str = None

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

    def is_quantizable(self, module: nn.Module) -> bool:
        if isinstance(module, nn.Conv2d) and self.quantize_conv:
            return True
        if isinstance(module, nn.Linear) and self.quantize_linear:
            return True
        return False

    def save_quantization_config(self, file_name: str):
        if self.cache_dir is None:
            return

        if os.path.exists(self.cache_dir) is False:
            os.makedirs(self.cache_dir)

        file_path = os.path.join(self.cache_dir, file_name)
        json.dump(self.__dict__, open(file_path, "w"))


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
        obj = [x.flatten() for x in obj]
        return torch.cat(obj)
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


class DefaultQuantizationCalibrator(QuantizationCalibratorInterface):
    def __init__(self, model: torch.nn.Module, config: QuantizationConfig):
        self.model = model
        self.config = config

    def calibrate(self, *args: Any, **kwargs: Any) -> Dict[str, str]:
        return {
            name: "1"
            for name, layer in self.model.named_modules()
            if self.config.is_quantizable(layer)
        }

    def plot_calibrate_info(self, *args: Any, **kwargs: Any):
        pass


class CostsQuantizationCalibrator(QuantizationCalibratorInterface):
    def __init__(
        self, model: torch.nn.Module, config: QuantizationConfig, cache_key: str = "",
    ):
        self.model = model
        self.config = config
        self.costs_file_name = f"costs_calibrate_info_{cache_key}.json"
        self.plot_calibrate_info_file_name = f"costs_calibrate_info_{cache_key}.html"

    def calibrate(self, *args: Any, **kwargs: Any) -> Dict[str, Dict[str, float]]:
        costs_calibrate_info = self.compute_quantization_costs(*args, **kwargs)
        filtered_costs_calibrate_info = {}
        for key, value in costs_calibrate_info.items():
            layer = get_sub_module(self.model, key)
            if (
                self.config.is_quantizable(layer)
                and value["compute_density"] >= self.config.compute_density_threshold
            ):
                filtered_costs_calibrate_info[key] = value

        if self.config.plot_calibrate_info and self.config.cache_dir is not None:
            file_path = os.path.join(self.config.cache_dir, self.plot_calibrate_info_file_name)
            self.plot_calibrate_info(
                file_path, list(costs_calibrate_info.items())
            )
            logger.info("Save Costs Calibrate Info to %s.html", self.costs_file_name)

        return filtered_costs_calibrate_info

    def plot_calibrate_info(self, file_path, info_lst: List):
        import plotly.graph_objs as go
        from plotly.offline import plot

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(info_lst))),
                y=[x[1]["compute_density"] for x in info_lst],
                mode="lines+markers",
                name="compute_density",
            )
        )
        fig.update_layout(title="Costs Calibrate Info")
        plot(fig, filename=file_path)

    def compute_quantization_costs(
        self, *args: Any, **kwargs: Any
    ) -> Dict[str, Dict[str, float]]:
        from onediff_quant.utils import metric_quantize_costs, find_quantizable_modules

        pipe = Pipe(self.model, args, kwargs)
        config = self.config
        quantizable_modules = find_quantizable_modules(
            self.model, module_cls=[nn.Linear, nn.Conv2d]
        )
        costs_calibrate_info: Dict[str, Dict[str, float]] = config.load_calibrate_info(
            self.costs_file_name
        )
        if len(costs_calibrate_info) == 0:
            costs = metric_quantize_costs(pipe, {}, quantizable_modules)
            costs_calibrate_info = {
                name: {"compute_density": costs.get_compute_density(name)}
                for name in quantizable_modules.keys()
            }
            config.save_calibrate_info(
                calibrate_info=costs_calibrate_info, file_name=self.costs_file_name
            )
            logger.info("Save Costs Calibrate Info to %s", self.costs_file_name)
        else:
            logger.info("Load Costs Calibrate Info from %s", self.costs_file_name)

        return costs_calibrate_info


class QuantizationMetricsCalculator(CostsQuantizationCalibrator):
    def __init__(
        self,
        model: torch.nn.Module,
        config: QuantizationConfig,
        cache_key: str,
        indicator_combination="111",
    ):
        super().__init__(model, config, cache_key)
        self.indicator_combination = indicator_combination
        self.linears_file_name = f"linear_layers_calibrate_info{cache_key}.json"
        self.convs_file_name = f"conv_layers_calibrate_info{cache_key}.json"
        self.plot_calibrate_info_file_name = (
            f"conv_linear_calibrate_info{cache_key}.html"
        )

    def calibrate_linear_layers(
        self, *args: Any, **kwargs: Any
    ) -> Dict[str, Dict[str, float]]:
        from onediff_quant.utils import find_quantizable_modules

        pipe = Pipe(self.model, args, kwargs)
        config = self.config
        quantizable_modules = find_quantizable_modules(
            self.model, module_cls=[nn.Linear]
        )
        calibrate_info: Dict[str, Dict[str, float]] = config.load_calibrate_info(
            self.linears_file_name
        )
        if len(calibrate_info) == 0 and config.quantize_linear:
            standard_output = unwrap_one(pipe(*args, **kwargs))
            calibrate_info = self._calibrate_modules(
                file_name=self.linears_file_name,
                modules=quantizable_modules,
                desc="Linear Layers",
                standard_output=standard_output,
                pipe=pipe,
                args=args,
                kwargs=kwargs,
            )
            config.save_calibrate_info(
                calibrate_info=calibrate_info, file_name=self.linears_file_name
            )
            logger.info(
                "Save Linear Layers Calibrate Info to %s", self.linears_file_name
            )
        else:
            logger.info(
                "Load Linear Layers Calibrate Info from %s", self.linears_file_name
            )
        return calibrate_info

    def calibrate_conv_layers(
        self, *args: Any, **kwargs: Any
    ) -> Dict[str, Dict[str, float]]:
        from onediff_quant.utils import find_quantizable_modules

        pipe = Pipe(self.model, args, kwargs)
        config = self.config
        quantizable_modules = find_quantizable_modules(
            self.model, module_cls=[nn.Conv2d]
        )
        calibrate_info: Dict[str, Dict[str, float]] = config.load_calibrate_info(
            self.convs_file_name
        )

        if len(calibrate_info) == 0 and config.quantize_conv:
            standard_output = unwrap_one(pipe(*args, **kwargs))
            calibrate_info = self._calibrate_modules(
                file_name=self.convs_file_name,
                modules=quantizable_modules,
                desc="Conv Layers",
                standard_output=standard_output,
                pipe=pipe,
                args=args,
                kwargs=kwargs,
            )
            config.save_calibrate_info(
                calibrate_info=calibrate_info, file_name=self.convs_file_name
            )
            logger.info("Save Conv Layers Calibrate Info to %s", self.convs_file_name)
        else:
            logger.info("Load Conv Layers Calibrate Info from %s", self.convs_file_name)

        return calibrate_info

    def _calibrate_modules(
        self,
        file_name: str,
        modules: Dict[str, nn.Module],
        desc: str,
        standard_output: torch.tensor,
        pipe: Pipe,
        args,
        kwargs,
    ) -> Dict[str, Dict[str, float]]:
        calibrate_info = {}

        for sub_name, layer in tqdm(modules.items(), desc=desc):
            handle = quantize_sub_module(
                model=pipe.model, sub_name=sub_name, sub_module=layer
            )
            current_output = unwrap_one(pipe(*args, **kwargs))
            mse = torch.mean((standard_output - current_output) ** 2)
            mae = torch.mean(torch.abs(standard_output - current_output))
            max_diff = torch.max(torch.abs(standard_output - current_output))
            calibrate_info[sub_name] = {
                "mse": mse.item(),
                "mae": mae.item(),
                "max_diff": max_diff.item(),
            }
            handle()

        self.config.save_calibrate_info(
            calibrate_info=calibrate_info, file_name=file_name
        )
        logger.info(f"Saved {desc} Calibrate Info to {file_name}")

        return calibrate_info

    @torch.no_grad()
    def calibrate(self, *args: Any, **kwargs: Any) -> Dict[str, Dict[str, float]]:
        costs_calibrate_info = self.compute_quantization_costs(*args, **kwargs)
        convs_calibrate_info = self.calibrate_conv_layers(*args, **kwargs)
        linears_calibrate_info = self.calibrate_linear_layers(*args, **kwargs)
        config = self.config
        all_calibrate_info = {**convs_calibrate_info, **linears_calibrate_info}

        quantizable_module_lst = [
            (name, layer)
            for name, layer in self.model.named_modules()
            if config.is_quantizable(layer)
        ]

        mae_lst, mse_lst, max_diff_lst = [], [], []
        linear_conv_names_lst = []

        for name, _ in quantizable_module_lst:
            mae_lst.append(all_calibrate_info[name]["mae"])
            mse_lst.append(all_calibrate_info[name]["mse"])
            max_diff_lst.append(all_calibrate_info[name]["max_diff"])
            linear_conv_names_lst.append(name)

        mae_lst, mse_lst, max_diff_lst = (
            normalize(mae_lst),
            normalize(mse_lst),
            normalize(max_diff_lst),
        )

        indexs = []
        if self.indicator_combination[0] == "1":
            indexs += find_peaks(mae_lst)
            # print(f"{len(find_peaks(mae_lst))=}")
        if self.indicator_combination[1] == "1":
            indexs += find_peaks(mse_lst)
            # print(f"{len(find_peaks(mse_lst))=}")
        if self.indicator_combination[2] == "1":
            indexs += find_peaks(max_diff_lst)
            # print(f"{len(find_peaks(max_diff_lst))=}")
        """
        sd_v1-5
        # len(find_peaks(mae_lst))=81
        # len(find_peaks(mse_lst))=36
        # len(find_peaks(max_diff_lst))=20
        """
        indexs = set(indexs)
        filtered_calibrate_info = {}
        compute_density_th = config.compute_density_threshold
        for i, (name, _) in enumerate(quantizable_module_lst):
            value = costs_calibrate_info[name]
            if i not in indexs and value["compute_density"] >= compute_density_th:
                filtered_calibrate_info[name] = value

        if config.plot_calibrate_info:
            info_lst = [
                (
                    name,
                    {"mse": mse_lst[i], "mae": mae_lst[i], "max_diff": max_diff_lst[i]},
                )
                for i, (name, _) in enumerate(quantizable_module_lst)
            ]
            file_name = self.plot_calibrate_info_file_name
            file_path = os.path.join(config.cache_dir, file_name)
            self.plot_calibrate_info(file_path=file_path, info_lst=info_lst)
        return filtered_calibrate_info

    def plot_calibrate_info(self, file_path, info_lst: List):
        import plotly.graph_objs as go
        from plotly.offline import plot

        fig = go.Figure()
        mse_list = [x[1]["mse"] for x in info_lst]
        mae_list = [x[1]["mae"] for x in info_lst]
        max_diff_list = [x[1]["max_diff"] for x in info_lst]
        x = list(range(len(info_lst)))
        y1, y2, y3 = normalize(mse_list), normalize(mae_list), normalize(max_diff_list)
        trace1 = go.Scatter(
            x=x, y=y1, mode="lines", name="mse", line=dict(color="blue")
        )
        trace2 = go.Scatter(x=x, y=y2, mode="lines", name="mae", line=dict(color="red"))
        trace3 = go.Scatter(
            x=x, y=y3, mode="lines", name="max_diff", line=dict(color="green")
        )
        layout = go.Layout(
            title="Linear Conv Calibrate Info",
            xaxis=dict(title="index"),
            yaxis=dict(title="value"),
        )
        data = [trace1, trace2, trace3]
        fig = go.Figure(data=data, layout=layout)
        plot(fig, filename=file_path)


class SubQuantizationPercentileCalculator(QuantizationMetricsCalculator):
    def __init__(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        cache_key: str,
        select="mae",
        conv_percentage=0.9,
        linear_percentage=0.9,
    ):
        assert select in ["mae", "mse", "max_diff"]
        super().__init__(
            model=model, config=config, cache_key=cache_key, indicator_combination="111"
        )
        self.select = select
        self.conv_percentage = conv_percentage
        self.linear_percentage = linear_percentage

    def calibrate(self, *args: Any, **kwargs: Any) -> Dict[str, Dict[str, float]]:
        costs_calibrate_info = self.compute_quantization_costs(*args, **kwargs)
        convs_calibrate_info = self.calibrate_conv_layers(*args, **kwargs)
        linears_calibrate_info = self.calibrate_linear_layers(*args, **kwargs)
        config = self.config
        convs_info_lst = [(name, value) for name, value in convs_calibrate_info.items()]
        linears_info_lst = [
            (name, value) for name, value in linears_calibrate_info.items()
        ]

        def get_quantizable_info_lst(info_lst, percentage):
            info_lst.sort(key=lambda x: x[1][self.select])
            return info_lst[: int(len(info_lst) * percentage)].copy()

        convs_info_lst = get_quantizable_info_lst(convs_info_lst, self.conv_percentage)
        linears_info_lst = get_quantizable_info_lst(
            linears_info_lst, self.linear_percentage
        )

        quantizable_module_lst = convs_info_lst + linears_info_lst
        filtered_costs_calibrate_info = {}
        compute_density_th = config.compute_density_threshold
        for name, _ in quantizable_module_lst:
            value = costs_calibrate_info[name]
            layer = get_sub_module(self.model, name)
            if (
                config.is_quantizable(layer)
                and value["compute_density"] >= compute_density_th
            ):
                filtered_costs_calibrate_info[name] = value

        return filtered_costs_calibrate_info


def create_quantization_calculator(
    model: torch.nn.Module,
    config: QuantizationConfig,
    cache_key: str,
    quality_level: int = 4,
):

    if quality_level == 4:
        return QuantizationMetricsCalculator(
            model=model, config=config, cache_key=cache_key, indicator_combination="100"
        )
    elif quality_level == 3:
        return QuantizationMetricsCalculator(
            model=model, config=config, cache_key=cache_key, indicator_combination="010"
        )
    elif quality_level == 2:
        return SubQuantizationPercentileCalculator(
            model=model,
            config=config,
            cache_key=cache_key,
            select="mae",
            conv_percentage=0.9,
            linear_percentage=0.9,
        )
    elif quality_level == 1:
        return CostsQuantizationCalibrator(model, config, cache_key)

    elif quality_level == 0:
        return DefaultQuantizationCalibrator(model, config)
    else:
        raise RuntimeWarning("Not Implemented")


class QuantizationModule:
    def __init__(
        self, torch_model: torch.nn.Module, quantization_config: QuantizationConfig,
    ):
        self.torch_model = torch_model
        self.quantization_config = quantization_config

    def quantize_forward(self, *args: Any, **kwargs: Any):
        if not self.quantization_config.use_quantization:
            return
        config = self.quantization_config
        basic_cache_key = (
            type(self.torch_model).__name__
            + "_"
            + calculate_model_hash(self.torch_model)
        )
        calculator = create_quantization_calculator(
            self.torch_model,
            config,
            basic_cache_key,
            quality_level=config.quality_level,
        )
        out = calculator.calibrate(*args, **kwargs)
        config.calibrate_info = out
        linear_names = [
            name
            for name, _ in out.items()
            if isinstance(get_sub_module(self.torch_model, name), nn.Linear)
        ]
        conv_names = [
            name
            for name, _ in out.items()
            if isinstance(get_sub_module(self.torch_model, name), nn.Conv2d)
        ]
        config.save_calibrate_info(
            f"{config.quality_level=}.json",
            calibrate_info={
                "linear_names_length": len(linear_names),
                "conv_names_length": len(conv_names),
            },
        )
