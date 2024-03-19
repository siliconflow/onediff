import types
import torch
from dataclasses import dataclass
from typing import Any, Dict
import oneflow as flow
from itertools import chain
from oneflow.utils.tensor import to_torch
from ...quantization.quantization_module import QuantizationConfig
from ..utils.log_utils import logger
from ..utils.cost_util import cost_cnt
from ..utils.args_tree_util import input_output_processor
from ..utils.oneflow_exec_mode import oneflow_exec_mode, oneflow_exec_mode_enabled
from ..utils.module_operations import get_sub_module
from ..utils.param_utils import parse_device, check_device
from ..utils.graph_management_utils import graph_file_management
from ..transform.manager import transform_mgr
from ..transform.builtin_transform import reverse_proxy_class, torch2oflow
from ...optimization.quant_optimizer import quantize_model

__all__ = ["OneFlowCompiledModel", "DualModule"]


class ParameterUpdateController:
    """Sync the parameter update between torch and oneflow module. """

    def __init__(self, dual_module: "DualModule", sync_with_oneflow_only=False):
        self.dual_module = dual_module
        self.sync_with_oneflow_only = sync_with_oneflow_only
        self.safe_module_id_map = {}
        self._synced = False
        self._handles = []

    def parameter_update(self, model_of: flow.nn.Module, key: str, value: Any) -> None:
        """ Update the parameters of the given model with the specified key-value pair."""
        # filter the training status
        if key == "training" or value is None:
            model_of.__dict__[key] = value
            return

        v = torch2oflow(value)
        if not self.dual_module._deployable_module_use_graph:
            setattr(model_of, key, v)
        elif isinstance(v, flow.Tensor):
            model_of.__dict__[key].copy_(v)
        elif isinstance(v, (int, float, bool, str)):
            if model_of.__dict__[key] == v:
                return
            model_of.__dict__[key] = v
            logger.warning(
                f"Only support oneflow.Tensor now. set {type(model_of)}.{key} = {v=}"
            )
        else:
            model_of.__dict__[key] = v
            logger.warning(
                f"Unsupported operation: Cannot set {type(model_of)}.{key} to {type(v)}"
            )

    def enable_sync(self):
        """
        Enable synchronization between PyTorch and OneFlow models.

        This method synchronizes the parameters of the PyTorch model with the corresponding parameters
        in the OneFlow model. It replaces the `__setattr__` method of each PyTorch module with a custom
        implementation that updates the corresponding OneFlow module whenever a parameter is set.
        """
        if self._synced or self.dual_module._oneflow_module is None:
            return
        self._synced = True

        logger.info(f"{'-'*20} Enable sync {'-'*20}")

        def _sync_torch_to_oneflow(model_pt, model_of, sub_module_name=""):
            org_setattr = model_pt.__class__.__setattr__

            def new_setattr(ins, name, value):
                nonlocal org_setattr, sub_module_name, self

                if id(model_of) == self.safe_module_id_map.get(id(ins), None):
                    self.parameter_update(model_of, name, value)
                else:
                    self.disable_sync()

                org_setattr(ins, name, value)

            model_pt.__class__.__setattr__ = new_setattr

            def restore():
                nonlocal org_setattr
                model_pt.__class__.__setattr__ = org_setattr

            self._handles.append(restore)

        torch_model, oneflow_model = self.dual_module.get_modules()
        self.safe_module_id_map.clear()
        for name, layer in torch_model.named_modules():
            model_of = get_sub_module(oneflow_model, name)
            key = id(layer)
            model_of_id = id(model_of)
            self.safe_module_id_map.setdefault(key, model_of_id)
            _sync_torch_to_oneflow(layer, model_of, name)

    def disable_sync(self):
        if not self._synced:
            return
        logger.info(f"{'-'*20} Disable sync {'-'*20}")
        self._synced = False
        for handle in self._handles:
            handle()
        self._handles.clear()

    def is_synced(self):
        return self._synced

    def __enter__(self):
        self.enable_sync()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.disable_sync()


@dataclass
class OneFlowCompiledModel:
    _torch_module: torch.nn.Module
    _oneflow_module: flow.nn.Module
    _parameter_update_controller: ParameterUpdateController
    _deployable_module_use_graph: bool
    _deployable_module_enable_dynamic: bool
    _deployable_module_options: Dict[str, Any]
    _deployable_module_dpl_graph: flow.nn.Graph
    _is_raw_deployable_module: bool
    _load_graph_first_run: bool


class DualModule(OneFlowCompiledModel):
    def __init__(self, torch_module, use_graph=True, dynamic=True, options={}):
        super().__init__(
            torch_module, None, None, use_graph, dynamic, options, None, False, True
        )
        self.quantization_config = QuantizationConfig()

    def get_modules(self):
        return self._torch_module, self._oneflow_module

    def get_graph_file(self):
        compiled_options = self._deployable_module_options
        return compiled_options.get("graph_file", None)

    def set_graph_file(self, file_path: str) -> None:
        old_file_path = self.get_graph_file()
        if old_file_path == file_path:
            return
        compiled_options = self._deployable_module_options
        compiled_options["graph_file"] = file_path
        self.clear_oneflow_module()

    @property  # Keep compatibility with previous changes.
    def _deployable_module_model(self):
        return self

    @property
    def oneflow_module(self):
        if self._oneflow_module is not None:
            return self._oneflow_module

        logger.debug(f"Convert {type(self._torch_module)} ...")
        if self.quantization_config.use_quantization:
            conf = self.quantization_config
            torch_module = quantize_model(
                self._torch_module,
                bits=conf.bits,
                quantize_conv=conf.quantize_conv,
                quantize_linear=conf.quantize_linear,
                inplace=conf.inplace,
                calibrate_info=conf.calibrate_info,
            )
            conf.use_quantization = False
        else:
            torch_module = self._torch_module

        self._oneflow_module = torch2oflow(torch_module)
        logger.debug(f"Convert {type(self._torch_module)} done!")
        self._parameter_update_controller = ParameterUpdateController(self, False)
        self._parameter_update_controller.enable_sync()
        return self._oneflow_module

    def get_graph(self):
        if self._deployable_module_dpl_graph is not None:
            return self._deployable_module_dpl_graph
        if "size" in self._deployable_module_options:
            size = self._deployable_module_options["size"]
        else:
            size = 9
        self._deployable_module_dpl_graph = get_oneflow_graph(
            self.oneflow_module, size, self._deployable_module_enable_dynamic,
        )
        # Enabel debug mode
        if transform_mgr.debug_mode:
            self._deployable_module_dpl_graph.debug(0)
        if "debug" in self._deployable_module_options:
            self._deployable_module_dpl_graph.debug(
                self._deployable_module_options["debug"]
            )
        return self._deployable_module_dpl_graph

    def clear_graph_cache(self):
        if self._deployable_module_dpl_graph is not None:
            del self._deployable_module_dpl_graph
            self._load_graph_first_run = True
            self._deployable_module_dpl_graph = None
            torch.cuda.empty_cache()
            flow.cuda.empty_cache()

    def clear_oneflow_module(self):
        self.clear_graph_cache()
        if self._oneflow_module is not None:
            del self._oneflow_module
            self._oneflow_module = None
            torch.cuda.empty_cache()
            flow.cuda.empty_cache()

    def disable_graph_file(self):
        self._deployable_module_options["graph_file"] = None

    @input_output_processor
    @graph_file_management
    def apply_model(self, *args, **kwargs):
        if self._deployable_module_use_graph:
            dpl_graph = self.get_graph()
            with oneflow_exec_mode():
                output = dpl_graph(*args, **kwargs)
        else:
            with oneflow_exec_mode():
                output = self.oneflow_module.apply_model(*args, **kwargs)

        return output

    @input_output_processor
    @graph_file_management
    def __call__(self, *args, **kwargs):
        if self._deployable_module_use_graph:
            dpl_graph = self.get_graph()

            with oneflow_exec_mode():
                output = dpl_graph(*args, **kwargs)
        else:
            with oneflow_exec_mode():
                output = self.oneflow_module(*args, **kwargs)
        return output

    @input_output_processor
    @graph_file_management
    def decode(self, *args, **kwargs):
        if self._deployable_module_use_graph:

            def _build(graph, *args, **kwargs):
                return graph.model.decode(*args, **kwargs)

            dpl_graph = self.get_graph()
            dpl_graph.build = types.MethodType(_build, dpl_graph)
            with oneflow_exec_mode():
                output = dpl_graph(*args, **kwargs)
        else:
            with oneflow_exec_mode():
                output = self.oneflow_module.decode(*args, **kwargs)
        return output

    def _to(self, *args, **kwargs):
        if oneflow_exec_mode_enabled():
            self._oneflow_module.to(*args, **kwargs)
        else:
            if self._oneflow_module is not None:
                of_args = [torch2oflow(v) for v in args]
                of_kwargs = {k: torch2oflow(v) for k, v in kwargs.items()}
                self._oneflow_module.to(*of_args, **of_kwargs)
                self._torch_module_to_with_check(*args, **kwargs)
            else:
                self._torch_module.to(*args, **kwargs)

    def to(self, *args, **kwargs):
        if self._deployable_module_dpl_graph is None:
            self._to(*args, **kwargs)
            return self

        target_device = parse_device(args, kwargs)
        if (
            target_device is not None
            and len(self._deployable_module_dpl_graph._blocks) > 0
        ):
            current_device = next(self._deployable_module_dpl_graph._state()).device
            if not check_device(current_device, target_device):
                raise RuntimeError(
                    f"After graph built, the device of graph can't be modified, current device: {current_device}, target device: {target_device}"
                )
        self._to(*args, **kwargs)
        return self

    def _torch_module_to_with_check(self, *args, **kwargs):
        def _align_tensor(torch_module, oneflow_module):
            oneflow_tensor_list = set(
                [x for x, _ in oneflow_module.named_parameters()]
                + [x for x, _ in oneflow_module.named_buffers()]
            )
            for name, tensor in chain.from_iterable(
                [torch_module.named_parameters(), torch_module.named_buffers(),]
            ):
                if name not in oneflow_tensor_list:
                    tensor.data = tensor.to(*args, **kwargs)
                else:
                    oneflow_tensor = oneflow_module.get_parameter(name)
                    if oneflow_tensor is None:
                        tensor.data = tensor.to(*args, **kwargs)
                    elif tensor.data_ptr() != oneflow_tensor.data_ptr():
                        tensor.data = to_torch(oneflow_tensor.data)

        oneflow_module_list = set([x for x, _ in self._oneflow_module.named_modules()])
        for name, module in self._torch_module.named_modules():
            if name not in oneflow_module_list:
                module.to(*args, **kwargs)
            else:
                _align_tensor(module, self._oneflow_module.get_submodule(name))

    def load_graph(self, file_path, device=None, run_warmup=True):
        self.get_graph().load_graph(file_path, device, run_warmup)

    def save_graph(self, file_path):
        self.get_graph().save_graph(file_path)


class OneflowGraph(flow.nn.Graph):
    @flow.nn.Graph.with_dynamic_input_shape()
    def __init__(self, model):
        super().__init__(enable_get_runtime_state_dict=True)
        self.model = model
        logger.info(f"Building a graph for {model.__class__.__name__} ...")
        # self.config.enable_cudnn_conv_heuristic_search_algo(False)
        self.config.allow_fuse_add_to_output(True)

    def build(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @cost_cnt(transform_mgr.debug_mode)
    def load_graph(self, file_path, device=None, run_warmup=True):
        state_dict = flow.load(file_path)
        if device is not None:
            state_dict = flow.nn.Graph.runtime_state_dict_to(state_dict, device)
        self.load_runtime_state_dict(state_dict, warmup_with_run=run_warmup)

    @cost_cnt(transform_mgr.debug_mode)
    def save_graph(self, file_path):
        state_dict = self.runtime_state_dict()

        import oneflow.framework.args_tree as args_tree

        def disabled_dataclass(value):
            return False

        original_is_dataclass = args_tree._is_dataclass
        args_tree._is_dataclass = disabled_dataclass

        import dataclasses

        def reverse_dataclass(value):
            if dataclasses.is_dataclass(value):
                return reverse_proxy_class(type(value))(**value)
            else:
                return value

        for name, rsd in state_dict.items():
            output = state_dict[name]["outputs_original"]
            out_tree = args_tree.ArgsTree((output, None), False)
            # dataclass type needs to be reversed to torch type to avoid saving error.
            out = out_tree.map_leaf(reverse_dataclass)
            state_dict[name]["outputs_original"] = out[0]

        args_tree._is_dataclass = original_is_dataclass
        flow.save(state_dict, file_path)


def get_oneflow_graph(model, size=9, dynamic_graph=True):
    g = OneflowGraph(model)
    g._dynamic_input_graph_cache.set_cache_size(size)
    g._dynamic_input_graph_cache.enable_shared(dynamic_graph)
    return g
