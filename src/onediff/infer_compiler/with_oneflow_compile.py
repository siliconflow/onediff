from typing import Any
from functools import wraps
from .torch_to_oflow.register import torch2oflow
import os
import types
import torch
import oneflow as flow
from .utils.oneflow_exec_mode import oneflow_exec_mode, oneflow_exec_mode_enabled
from .utils.args_tree_util import input_output_processor
from .registry import set_default_registry


class DualModule(torch.nn.Module):
    def __init__(self, torch_module, oneflow_module):
        super().__init__()
        self._torch_module = torch_module
        self._oneflow_module = oneflow_module

    @property
    def oneflow_module(self):
        if self._oneflow_module is not None:
            return self._oneflow_module
        self._oneflow_module = torch2oflow(self._torch_module)
        return self._oneflow_module

    @oneflow_module.deleter
    def oneflow_module(self):
        if self._oneflow_module:
            del self._oneflow_module
            setattr(self, "_oneflow_module", None)

    def to(self, *args, **kwargs):
        if oneflow_exec_mode_enabled():
            self._oneflow_module.to(*args, **kwargs)
        else:
            if self._oneflow_module is not None:
                args = [torch2oflow(v) for v in args]
                kwargs = {k: torch2oflow(v) for k, v in kwargs.items()}
                self._oneflow_module.to(*args, **kwargs)
            else:
                self._torch_module.to(*args, **kwargs)

    def __getattr__(self, name):
        if name == "_torch_module":
            return self._modules[name]
        if name == "_oneflow_module":
            return super().__getattribute__(name)

        torch_attr = getattr(self._torch_module, name)
        oneflow_attr = (
            None
            if self._oneflow_module is None
            else getattr(self._oneflow_module, name)
        )
        if isinstance(torch_attr, torch.nn.Module):
            return DualModule(torch_attr, oneflow_attr)
        else:
            return oneflow_attr if oneflow_exec_mode_enabled() else torch_attr

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ["_torch_module", "_oneflow_module"]:
            super().__setattr__(name, value)
        else:  # TODO: aviod memory up when set attr
            if self._oneflow_module is not None:
                obj = getattr(self._oneflow_module, name)
                obj.copy_(torch2oflow(value))
            else:
                setattr(self._torch_module, name, value)


def handle_deployable_exception(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            print(
                f"Exception in {func.__name__} of " f"{self.__class__.__name__}: {e=}"
            )
            print("Recompile oneflow module ...")
            del self._deployable_module_model.oneflow_module
            self._deployable_module_dpl_graph = None
            return func(self, *args, **kwargs)

    return wrapper


class DeployableModule(torch.nn.Module):
    def __init__(self, torch_module, oneflow_module, use_graph=True, options={}):
        super().__init__()
        self._deployable_module_model = DualModule(torch_module, oneflow_module)
        self._deployable_module_use_graph = use_graph
        self._deployable_module_options = options
        self._deployable_module_dpl_graph = None

    @classmethod
    def from_existing(cls, existing_module, use_graph=None, options=None):
        torch_module = existing_module._deployable_module_model._torch_module
        oneflow_module = existing_module._deployable_module_model._oneflow_module
        instance = cls(torch_module, oneflow_module, use_graph, options)
        instance._deployable_module_dpl_graph = (
            existing_module._deployable_module_dpl_graph if use_graph else None
        )
        return instance

    def get_graph(self):
        if self._deployable_module_dpl_graph is not None:
            return self._deployable_module_dpl_graph
        if "size" in self._deployable_module_options:
            size = self._deployable_module_options["size"]
        else:
            size = 9
        self._deployable_module_dpl_graph = get_oneflow_graph(
            self._deployable_module_model.oneflow_module, size
        )
        return self._deployable_module_dpl_graph

    @input_output_processor
    @handle_deployable_exception
    def apply_model(self, *args, **kwargs):
        if self._deployable_module_use_graph:
            dpl_graph = self.get_graph()
            with oneflow_exec_mode():
                output = dpl_graph(*args, **kwargs)
        else:
            with oneflow_exec_mode():
                output = self._deployable_module_model.oneflow_module.apply_model(
                    *args, **kwargs
                )
        return output

    @input_output_processor
    @handle_deployable_exception
    def __call__(self, *args, **kwargs):
        if self._deployable_module_use_graph:
            dpl_graph = self.get_graph()
            with oneflow_exec_mode():
                output = dpl_graph(*args, **kwargs)
        else:
            with oneflow_exec_mode():
                output = self._deployable_module_model.oneflow_module(*args, **kwargs)
        return output

    def to(self, *args, **kwargs):
        self._deployable_module_model.to(*args, **kwargs)
        return self

    # TODO(): Just for transformers VAE decoder
    @input_output_processor
    @handle_deployable_exception
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
                output = self._deployable_module_model.oneflow_module.decode(
                    *args, **kwargs
                )
        return output

    def __getattr__(self, name):
        if name in self._modules:
            return self._modules[name]
        return getattr(self._deployable_module_model, name)

    def load_graph(self, file_path, device=None):
        self.get_graph().warmup_with_load(file_path, device)

    def warmup_with_load(self, file_path, device=None):
        self.get_graph().warmup_with_load(file_path, device)

    def save_graph(self, file_path):
        self.get_graph().save_graph(file_path)


class OneflowGraph(flow.nn.Graph):
    @flow.nn.Graph.with_dynamic_input_shape()
    def __init__(self, model):
        super().__init__(enable_get_runtime_state_dict=True)
        self.model = model
        self.config.enable_cudnn_conv_heuristic_search_algo(False)
        self.config.allow_fuse_add_to_output(True)

        os.environ["ONEFLOW_GRAPH_DELAY_VARIABLE_OP_EXECUTION"] = "1"
        os.environ["ONEFLOW_MLIR_CSE"] = "1"
        os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
        os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
        os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
        os.environ["ONEFLOW_MLIR_FUSE_OPS_WITH_BACKWARD_IMPL"] = "1"
        os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
        os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"
        os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS"] = "1"
        os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"
        os.environ["ONEFLOW_KERNEL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"
        os.environ["ONEFLOW_KERNEL_GEMM_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"
        os.environ["ONEFLOW_KERNEL_CONV_ENABLE_CUTLASS_IMPL"] = "1"
        os.environ["ONEFLOW_KERNEL_GEMM_ENABLE_CUTLASS_IMPL"] = "1"
        os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
        os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
        os.environ["ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT"] = "1"
        os.environ["ONEFLOW_KERNEL_GLU_ENABLE_DUAL_GEMM_IMPL"] = "0"
        os.environ["ONEFLOW_MLIR_GROUP_MATMUL_QUANT"] = "1"
        # TODO: enable this will cause the failure of multi resolution warmup
        # os.environ["ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH"] = "1"
        # os.environ["ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH"] = "1"

    def build(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def warmup_with_load(self, file_path, device=None):
        state_dict = flow.load(file_path)
        if device is not None:
            state_dict = flow.nn.Graph.runtime_state_dict_to(state_dict, device)
        self.load_runtime_state_dict(state_dict)

    def save_graph(self, file_path):
        state_dict = self.runtime_state_dict()
        flow.save(state_dict, file_path)


def get_oneflow_graph(model, size=9):
    g = OneflowGraph(model)
    g._dynamic_input_graph_cache.set_cache_size(size)
    return g


def state_dict_hook(module, state_dict, prefix, local_metadata):
    pytorch_key_prefix = "_deployable_module_model._torch_module."
    new_state_dict = type(state_dict)()
    for k, v in state_dict.items():
        # key_filter
        # _deployable_module_model._torch_module.out.2.weight => out.2.weight
        if k.startswith(pytorch_key_prefix):
            new_k = k[len(pytorch_key_prefix) :]
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def oneflow_compile(torch_module, *, use_graph=True, options={}):
    set_default_registry()

    def wrap_module(module):
        if isinstance(module, DeployableModule):
            return DeployableModule.from_existing(module, use_graph, options)
        else:
            return DeployableModule(module, None, use_graph, options)

    model = wrap_module(torch_module)
    model._register_state_dict_hook(state_dict_hook)
    return model
