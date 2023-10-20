import types
from .convert_torch_to_of.register import torch2of
import os
import torch
import oneflow as flow
from oneflow.framework.args_tree import ArgsTree

from .utils import (
    oneflow_exec_mode,
    oneflow_exec_mode_enabled,
    register_args_tree_relaxed_types,
)

register_args_tree_relaxed_types()


class DualModule(torch.nn.Module):
    def __init__(self, torch_module, oneflow_module):
        super().__init__()
        self._torch_module = torch_module
        self._oneflow_module = oneflow_module

    def to(self, *args, **kwargs):
        if oneflow_exec_mode_enabled():
            self._oneflow_module.to(*args, **kwargs)
        else:
            self._torch_module.to(*args, **kwargs)
            args = [torch2of(v) for v in args]
            kwargs = {k: torch2of(v) for k, v in kwargs.items()}
            self._oneflow_module.to(*args, **kwargs)

    def __getattr__(self, name):
        if name == "_torch_module":
            return self._modules[name]
        if name == "_oneflow_module":
            return super().__getattribute__(name)

        torch_attr = getattr(self._torch_module, name)
        oneflow_attr = getattr(self._oneflow_module, name)
        if isinstance(torch_attr, torch.nn.ModuleList):
            return DualModuleList(torch_attr, oneflow_attr)
        elif isinstance(torch_attr, torch.nn.Module):
            return DualModule(torch_attr, oneflow_attr)
        else:
            return oneflow_attr if oneflow_exec_mode_enabled() else torch_attr


class DualModuleList(torch.nn.ModuleList):
    def __init__(self, torch_module, oneflow_module):
        super().__init__()
        self.torch_module = torch_module
        self.oneflow_module = oneflow_module

    def __getitem__(self, idx):
        return DualModule(self.torch_module[idx], self.oneflow_module[idx])


class DeployableModule(torch.nn.Module):
    def __init__(self, torch_module, oneflow_module, use_graph=True, options={}):
        super().__init__()
        self._deployable_module_model = DualModule(torch_module, oneflow_module)
        self._deployable_module_use_graph = use_graph
        self._deployable_module_options = options
        self._deployable_module_dpl_graph = None

    def process_input(self, *args, **kwargs):
        def input_fn(value):
            if isinstance(value, torch.Tensor):
                return flow.utils.tensor.from_torch(value)
            else:
                return value

        args_tree = ArgsTree((args, kwargs), False, tensor_type=torch.Tensor)
        out = args_tree.map_leaf(input_fn)
        mapped_args = out[0]
        mapped_kwargs = out[1]
        return mapped_args, mapped_kwargs

    def process_output(self, output):
        def output_fn(value):
            if isinstance(value, flow.Tensor):
                return flow.utils.tensor.to_torch(value)
            else:
                return value

        out_tree = ArgsTree((output, None), False)
        out = out_tree.map_leaf(output_fn)
        return out[0]

    def get_graph(self):
        if self._deployable_module_dpl_graph is not None:
            return self._deployable_module_dpl_graph
        if "size" in self._deployable_module_options:
            size = self._deployable_module_options["size"]
        else:
            size = 9
        self._deployable_module_dpl_graph = get_oneflow_graph(size)(
            self._deployable_module_model._oneflow_module
        )
        return self._deployable_module_dpl_graph

    def apply_model(self, *args, **kwargs):
        mapped_args, mapped_kwargs = self.process_input(*args, **kwargs)
        if self._deployable_module_use_graph:
            dpl_graph = self.get_graph()
            with oneflow_exec_mode():
                output = dpl_graph(*mapped_args, **mapped_kwargs)
        else:
            with oneflow_exec_mode():
                output = self._deployable_module_model._oneflow_module.apply_model(
                    *mapped_args, **mapped_kwargs
                )
        return self.process_output(output)

    def to(self, *args, **kwargs):
        self._deployable_module_model.to(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        mapped_args, mapped_kwargs = self.process_input(*args, **kwargs)
        if self._deployable_module_use_graph:
            dpl_graph = self.get_graph()
            with oneflow_exec_mode():
                output = dpl_graph(*mapped_args, **mapped_kwargs)
        else:
            with oneflow_exec_mode():
                output = self._deployable_module_model._oneflow_module(
                    *mapped_args, **mapped_kwargs
                )
        return self.process_output(output)

    # TODO(): Just for transformers VAE decoder
    def decode(self, *args, **kwargs):
        mapped_args, mapped_kwargs = self.process_input(*args, **kwargs)
        if self._deployable_module_use_graph:

            def _build(graph, *args, **kwargs):
                return graph.model.decode(*args, **kwargs)

            dpl_graph = self.get_graph()
            dpl_graph.build = types.MethodType(_build, dpl_graph)
            with oneflow_exec_mode():
                output = dpl_graph(*mapped_args, **mapped_kwargs)
        else:
            with oneflow_exec_mode():
                output = self._deployable_module_model._oneflow_module.decode(
                    *mapped_args, **mapped_kwargs
                )
        return self.process_output(output)

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


def get_oneflow_graph(size=9):
    class OneflowGraph(flow.nn.Graph):
        @flow.nn.Graph.with_dynamic_input_shape(size=size) 
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

    return OneflowGraph


def oneflow_compile(torch_module, *, use_graph=True, options={}):
    oneflow_module = torch2of(torch_module)
    return DeployableModule(torch_module, oneflow_module, use_graph, options)

# TODO() model_patcher https://github.com/siliconflow/comfyui-speedup/blob/ad8b2d4b31272543f97aef80cb92ae22d88066ae/nodes.py#L25
def oneflow_compile_lazy(torch_module, *, use_graph=True, options={}):
    """Lazy compilation of torch module to oneflow module.

    Calling the `__call__` method of LazyOneFlowModule will trigger compilation.

    Example:
        >>> import torch
        >>> from onediff.infer_compiler import oneflow_compile_lazy
        >>> torch_module = torch.nn.Linear(3, 4)
        >>> oneflow_module = oneflow_compile_lazy(torch_module, use_graph=True)
        >>> torch_module.to("cuda")
        >>> input = torch.randn(2, 3).to("cuda")
        >>> oneflow_module(input)
        
        oneflow_compile_lazy __call__ ...

        tensor([[-0.6069, -0.5079, -0.1984, -0.1253],
                [ 0.0041, -0.0595,  0.1333, -0.4581]], device='cuda:0')
    """

    class LazyOneFlowModule:
        def __init__(self, torch_module):
            self._torch_module = torch_module
            self._lazy_convert = True
            self._oneflow_module = None

        def __getattribute__(self, __name: str):
            if __name in ("_torch_module", "_lazy_convert", "_oneflow_module"):
                return super().__getattribute__(__name)
            if self._lazy_convert:
                return getattr(self._torch_module, __name)
            else:
                return getattr(self._oneflow_module, __name)

        def __call__(self, *args, **kwargs):
            if self._oneflow_module is None:
                print("oneflow_compile_lazy __call__ ...")
                self._oneflow_module = oneflow_compile(
                    self._torch_module, use_graph=use_graph, options=options
                )
                self._lazy_convert = False
            return self._oneflow_module(*args, **kwargs)

    return LazyOneFlowModule(torch_module)
