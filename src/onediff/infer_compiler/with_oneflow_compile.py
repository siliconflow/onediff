from .convert_torch_to_of.register import torch2of
import os
import torch
import oneflow as flow

from .utils import oneflow_graph_mode, oneflow_graph_mode_enabled


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


@torch.no_grad()
def oneflow_compile(torch_module, *, use_graph=True, options={}):

    oneflow_module = torch2of(torch_module)

    from oneflow.framework.args_tree import ArgsTree

    def input_fn(value):
        if isinstance(value, torch.Tensor):
            return flow.utils.tensor.from_torch(value)
        else:
            return value

    def output_fn(value):
        if isinstance(value, flow.Tensor):
            return flow.utils.tensor.to_torch(value)
        else:
            return value

    if use_graph:
        if "size" in options:
            size = options["size"]
        else:
            size = 9
        dpl_graph = get_oneflow_graph(size)(oneflow_module)

    class DeployableModule(torch.nn.Module):
        def __init__(self, torch_module, oneflow_module):
            super().__init__()
            self._torch_module = torch_module
            self._oneflow_module = oneflow_module

        def process_input(self, *args, **kwargs):
            args_tree = ArgsTree((args, kwargs), False, tensor_type=torch.Tensor)
            out = args_tree.map_leaf(input_fn)
            mapped_args = out[0]
            mapped_kwargs = out[1]
            return mapped_args, mapped_kwargs

        def process_output(self, output):
            out_tree = ArgsTree((output, None), False)
            out = out_tree.map_leaf(output_fn)
            return out[0]

        def apply_model(self, *args, **kwargs):
            if hasattr(super(), "apply_model"):
                mapped_args, mapped_kwargs = self.process_input(*args, **kwargs)
                if use_graph:
                    output = self._oneflow_module._dpl_graph(
                        *mapped_args, **mapped_kwargs
                    )
                else:
                    output = self._oneflow_module.apply_model(
                        *mapped_args, **mapped_kwargs
                    )
                return self.process_output(output)

        def __call__(self, *args, **kwargs):
            with oneflow_graph_mode():
                mapped_args, mapped_kwargs = self.process_input(*args, **kwargs)
                if use_graph:
                    output = self._oneflow_module._dpl_graph(
                        *mapped_args, **mapped_kwargs
                    )
                else:
                    output = self._oneflow_module(*mapped_args, **mapped_kwargs)
            return self.process_output(output)

        def __getattr__(self, name):
            if name == "_torch_module":
                return self._modules[name]
            if name == "_oneflow_module":
                return super().__getattribute__(name)
            if oneflow_graph_mode_enabled():
                return getattr(self._oneflow_module, name)
            else:
                return getattr(self._torch_module, name)

        def load_graph(self, file_path, device=None):
            self._oneflow_module._dpl_graph.warmup_with_load(file_path, device)

        def warmup_with_load(self, file_path, device=None):
            self._oneflow_module._dpl_graph.warmup_with_load(file_path, device)

        def save_graph(self, file_path):
            self._oneflow_module._dpl_graph.save_graph(file_path)

    if use_graph:
        oneflow_module._dpl_graph = dpl_graph
    return DeployableModule(torch_module, oneflow_module)
