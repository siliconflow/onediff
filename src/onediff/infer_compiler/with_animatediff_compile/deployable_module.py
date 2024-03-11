import torch
import torch.nn as nn
from ..utils.log_utils import logger
from ..utils.version_util import is_community_version
from .oneflow_compiler import QuantizationConfig, DualModule


class DeployableModule:
    """
    A wrapper class for a compiled model.
    """

    __attrs = [
        "_deployed_original_module",
        "_deployed_compiled_model",
        "quantization_config",
    ]

    def __init__(self, original_module: torch.nn.Module, compiled_model: DualModule=None):
        assert compiled_model is not None, "Compiled model is None"
        self._deployed_original_module = original_module
        self._deployed_compiled_model = compiled_model
        self.quantization_config: QuantizationConfig = self._deployed_compiled_model.quantization_config

    def quantize(
        self,
        quantize_conv=True,
        quantize_linear=True,
        bits=8,
        compute_density_threshold=10,
        *,
        inplace=True,
        calibrate_info=None,
        cache_dir=None,
    ):
        if is_community_version():
            logger.warning("quantize is not supported in community version")
            return
        config = self.quantization_config
        config.__dict__.update(
            {
                "quantize_conv": quantize_conv,
                "quantize_linear": quantize_linear,
                "bits": bits,
                "inplace": inplace,
                "compute_density_threshold": compute_density_threshold,
                "calibrate_info": calibrate_info,
                "use_quantization": True,
                "cache_dir": cache_dir,
            }
        )
        return self

    @property  # Keep compatibility with previous changes.
    def _deployable_module_model(self):
        return self._deployed_compiled_model

    @property  # Keep compatibility with previous changes.
    def _torch_module(self):
        return self._deployed_original_module

    @property
    def __class__(self):
        return self._deployed_original_module.__class__

    def set_graph_file(self, file_path: str):
        self._deployed_compiled_model.set_graph_file(file_path)

    def get_graph_file(self):
        """
        Returns:
            str: The path to the graph file.
        """
        return self._deployed_compiled_model.get_graph_file()

    def disalbe_graph_file(self):
        self._deployed_compiled_model.disable_graph_file()

    def get_graph(self):
        return self._deployed_compiled_model.get_graph()

    def load_graph(self, file_path, device=None, run_warmup=True):
        self._deployed_compiled_model.load_graph(file_path, device, run_warmup)

    def save_graph(self, file_path):
        self._deployed_compiled_model.save_graph(file_path)

    def decode(self, *args, **kwargs):
        return self._deployed_compiled_model.decode(*args, **kwargs)

    def to(self, *args, **kwargs):
        return self._deployed_compiled_model.to(*args, **kwargs)

    def _quantize_forward_faster(self, *args, **kwargs):
        if not self.quantization_config.use_quantization:
            return
        from onediff_quant.utils import find_quantizable_modules
        from onediff_quant.utils import metric_quantize_costs

        class Pipe:
            def __init__(self, model, args, kwargs):
                self.model = model
                self.args = args
                self.kwargs = kwargs

            def __call__(self, *args, **kwds):
                return self.model(*self.args, **self.kwargs)

        pipe = Pipe(self._deployed_original_module, args, kwargs)

        module_cls = []
        if self.quantization_config.quantize_conv:
            module_cls.append(nn.Conv2d)
        if self.quantization_config.quantize_linear:
            module_cls.append(nn.Linear)

        quantizable_modules = find_quantizable_modules(
            pipe.model, module_cls=module_cls
        )

        costs = metric_quantize_costs(pipe, {}, quantizable_modules)
        compute_density_threshold = self.quantization_config.compute_density_threshold
        calibrate_info = {}
        for name, layer in quantizable_modules.items():
            if (
                isinstance(layer, nn.Linear)
                and not self.quantization_config.quantize_linear
            ):
                continue
            if (
                isinstance(layer, nn.Conv2d)
                and not self.quantization_config.quantize_conv
            ):
                continue
            if costs.get_compute_density(name) > compute_density_threshold:
                calibrate_info[name] = "1"

        self.quantization_config.calibrate_info = calibrate_info

    def __call__(self, *args, **kwargs):
        if (
            self.quantization_config.use_quantization
            and self.quantization_config.calibrate_info is None
        ):
            self._quantize_forward_faster(*args, **kwargs)
        out = self._deployed_compiled_model(*args, **kwargs)
        return out

    def __getattr__(self, name):
        if name in DeployableModule.__attrs:
            return super().__getattribute__(self, name)
        return getattr(self._deployed_original_module, name)

    def __setattr__(self, name, value):
        if name in DeployableModule.__attrs:
            return super().__setattr__(name, value)
        setattr(self._deployed_original_module, name, value)

    def __getitem__(self, key):
        return self._deployed_original_module[key]

    def __setitem__(self, key, value):
        self._deployed_original_module[key] = value

    def __delitem__(self, key):
        del self._deployed_original_module[key]

    def __dir__(self):
        return dir(self._deployed_original_module)

    def __repr__(self):
        return f"DeployableModule({self._deployed_original_module})"
