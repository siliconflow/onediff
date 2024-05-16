import dataclasses
from typing import Dict
import torch


@dataclasses.dataclass
class OneflowCompileOptions:
    use_graph: bool = True
    debug_level: int = -1
    max_cached_graph_size: int = 9
    graph_file: str = None
    graph_file_device: torch.device = None

    # Optimization related environment variables
    run_graph_by_vm: bool = None
    graph_delay_variable_op_execution: bool = None

    conv_allow_half_precision_accumulation: bool = None
    matmul_allow_half_precision_accumulation: bool = None
    attention_allow_half_precision_accumulation: bool = None
    attention_allow_half_precision_score_accumulation_max_m: int = None
    attention_allow_quantization: bool = None

    mlir_cse: bool = None
    mlir_enable_inference_optimization: bool = None
    mlir_enable_round_trip: bool = None
    mlir_fuse_forward_ops: bool = None
    mlir_fuse_ops_with_backward_impl: bool = None
    mlir_group_matmul: bool = None
    mlir_prefer_nhwc: bool = None
    mlir_fuse_kernel_launch: bool = None

    kernel_enable_cuda_graph: bool = None
    kernel_enable_fused_conv_bias: bool = None
    kernel_enable_fused_linear: bool = None
    kernel_conv_cutlass_impl_enable_tuning_warmup: bool = None
    kernel_enable_conv2d_tuning_warmup: bool = None
    kernel_gemm_cutlass_impl_enable_tuning_warmup: bool = None
    kernel_conv_enable_cutlass_impl: bool = None
    kernel_gemm_enable_cutlass_impl: bool = None
    kernel_glu_enable_dual_gemm_impl: bool = None
    kernel_glu_enable_y_gemm_impl: bool = None
    kernel_glu_quant_enable_dual_gemm_impl: bool = None

@dataclasses.dataclass
class CompileOptions:
    # common options
    dynamic: bool

    # oneflow specific options
    oneflow: OneflowCompileOptions

    # nexfort specific options
    nexfort: Dict

    def __init__(self, dynamic=True, oneflow=None, nexfort=None):
        self.dynamic = dynamic
        self.oneflow = oneflow if oneflow is not None else OneflowCompileOptions()
        self.nexfort = nexfort if nexfort is not None else dict()


# a global default compile options
_GLOBAL_compile_options = CompileOptions()
