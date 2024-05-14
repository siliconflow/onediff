import os
from typing import Optional
import dataclasses
from ..utils import (
    parse_boolean_from_env,
    set_boolean_env_var,
    parse_integer_from_env,
    set_integer_env_var,
)


def init_default_env():
    # ONEFLOW_RUN_GRAPH_BY_VM must set here to enable nn.Graph init with vm run
    os.environ.setdefault("ONEFLOW_RUN_GRAPH_BY_VM", "1")
    os.environ.setdefault("ONEFLOW_GRAPH_DELAY_VARIABLE_OP_EXECUTION", "1")

    os.environ.setdefault("ONEFLOW_MLIR_CSE", "1")
    os.environ.setdefault("ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION", "1")
    os.environ.setdefault("ONEFLOW_MLIR_ENABLE_ROUND_TRIP", "1")
    os.environ.setdefault("ONEFLOW_MLIR_FUSE_FORWARD_OPS", "1")
    os.environ.setdefault("ONEFLOW_MLIR_FUSE_OPS_WITH_BACKWARD_IMPL", "1")
    os.environ.setdefault("ONEFLOW_MLIR_GROUP_MATMUL", "1")
    os.environ.setdefault("ONEFLOW_MLIR_PREFER_NHWC", "1")

    os.environ.setdefault("ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS", "1")
    os.environ.setdefault("ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR", "1")
    os.environ.setdefault("ONEFLOW_KERNEL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP", "1")
    os.environ.setdefault("ONEFLOW_KERNEL_GEMM_CUTLASS_IMPL_ENABLE_TUNING_WARMUP", "1")
    os.environ.setdefault("ONEFLOW_KERNEL_CONV_ENABLE_CUTLASS_IMPL", "1")
    os.environ.setdefault("ONEFLOW_KERNEL_GEMM_ENABLE_CUTLASS_IMPL", "1")
    os.environ.setdefault("ONEFLOW_CONVOLUTION_BIAS_ADD_ACT_FUSION", "1")
    # os.environ.setdefault("ONEFLOW_KERNEL_GLU_ENABLE_DUAL_GEMM_IMPL", "0")
    # os.environ.setdefault("ONEFLOW_KERNEL_GLU_ENABLE_Y_GEMM_IMPL", "0")
    # os.environ.setdefault("ONEFLOW_KERNEL_GLU_QUANT_ENABLE_DUAL_GEMM_IMPL", "0")

    os.environ.setdefault("ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION", "1")
    os.environ.setdefault("ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION", "1")
    os.environ.setdefault("ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT", "1")
    # os.environ.setdefault("ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_ACCUMULATION", "1")
    # os.environ.setdefault("ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_SCORE_ACCUMULATION_MAX_M", "-1")
    # os.environ.setdefault("ONEFLOW_ATTENTION_ALLOW_QUANTIZATION", "1")

    os.environ.setdefault("ONEFLOW_MLIR_GROUP_MATMUL_QUANT", "1")
    os.environ.setdefault("ONEFLOW_CONV2D_KERNEL_ENABLE_TUNING_WARMUP", "1")
    # TODO: enable this will cause the failure of multi resolution warmup
    # os.environ.setdefault("ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH", "1")
    # os.environ.setdefault("ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH", "1")


@dataclasses.dataclass
class OneFlowCompilerConfig:
    run_graph_by_vm: Optional[bool] = None
    graph_delay_variable_op_execution: Optional[bool] = None

    mlir_cse: Optional[bool] = None
    mlir_enable_inference_optimization: Optional[bool] = None
    mlir_enable_round_trip: Optional[bool] = None
    mlir_fuse_forward_ops: Optional[bool] = None
    mlir_fuse_ops_with_backward_impl: Optional[bool] = None
    mlir_group_matmul: Optional[bool] = None
    mlir_prefer_nhwc: Optional[bool] = None
    mlir_fuse_kernel_launch: Optional[bool] = None

    kernel_enable_cuda_graph: Optional[bool] = None
    kernel_enable_fused_conv_bias: Optional[bool] = None
    kernel_enable_fused_linear: Optional[bool] = None
    kernel_conv_cutlass_impl_enable_tuning_warmup: Optional[bool] = None
    kernel_gemm_cutlass_impl_enable_tuning_warmup: Optional[bool] = None
    kernel_conv_enable_cutlass_impl: Optional[bool] = None
    kernel_gemm_enable_cutlass_impl: Optional[bool] = None
    kernel_glu_enable_dual_gemm_impl: Optional[bool] = None
    kernel_glu_enable_y_gemm_impl: Optional[bool] = None
    kernel_glu_quant_enable_dual_gemm_impl: Optional[bool] = None

    conv_allow_half_precision_accumulation: Optional[bool] = None
    matmul_allow_half_precision_accumulation: Optional[bool] = None
    linear_embedding_skip_init: Optional[bool] = None
    attention_allow_half_precision_accumulation: Optional[bool] = None
    attention_allow_half_precision_score_accumulation_max_m: Optional[int] = None
    attention_allow_quantization: Optional[bool] = None
    conv2d_kernel_enable_tuning_warmup: Optional[bool] = None

    attr2env_var = {
        "run_graph_by_vm": "ONEFLOW_RUN_GRAPH_BY_VM",
        "graph_delay_variable_op_execution": "ONEFLOW_GRAPH_DELAY_VARIABLE_OP_EXECUTION",
        "mlir_cse": "ONEFLOW_MLIR_CSE",
        "mlir_enable_inference_optimization": "ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION",
        "mlir_enable_round_trip": "ONEFLOW_MLIR_ENABLE_ROUND_TRIP",
        "mlir_fuse_forward_ops": "ONEFLOW_MLIR_FUSE_FORWARD_OPS",
        "mlir_fuse_ops_with_backward_impl": "ONEFLOW_MLIR_FUSE_OPS_WITH_BACKWARD_IMPL",
        "mlir_group_matmul": "ONEFLOW_MLIR_GROUP_MATMUL",
        "mlir_prefer_nhwc": "ONEFLOW_MLIR_PREFER_NHWC",
        "mlir_fuse_kernel_launch": "ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH",
        "kernel_enable_cuda_graph": "ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH",
        "kernel_enable_fused_conv_bias": "ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS",
        "kernel_enable_fused_linear": "ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR",
        "kernel_conv_cutlass_impl_enable_tuning_warmup": "ONEFLOW_KERNEL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP",
        "kernel_gemm_cutlass_impl_enable_tuning_warmup": "ONEFLOW_KERNEL_GEMM_CUTLASS_IMPL_ENABLE_TUNING_WARMUP",
        "kernel_conv_enable_cutlass_impl": "ONEFLOW_KERNEL_CONV_ENABLE_CUTLASS_IMPL",
        "kernel_gemm_enable_cutlass_impl": "ONEFLOW_KERNEL_GEMM_ENABLE_CUTLASS_IMPL",
        "kernel_glu_enable_dual_gemm_impl": "ONEFLOW_KERNEL_GLU_ENABLE_DUAL_GEMM_IMPL",
        "kernel_glu_enable_y_gemm_impl": "ONEFLOW_KERNEL_GLU_ENABLE_Y_GEMM_IMPL",
        "kernel_glu_quant_enable_dual_gemm_impl": "ONEFLOW_KERNEL_GLU_QUANT_ENABLE_DUAL_GEMM_IMPL",
        "conv_allow_half_precision_accumulation": "ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION",
        "matmul_allow_half_precision_accumulation": "ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION",
        "linear_embedding_skip_init": "ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT",
        "attention_allow_half_precision_accumulation": "ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_ACCUMULATION",
        "attention_allow_half_precision_score_accumulation_max_m": "ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_SCORE_ACCUMULATION_MAX_M",
        "conv2d_kernel_enable_tuning_warmup":'ONEFLOW_CONV2D_KERNEL_ENABLE_TUNING_WARMUP',
    }

    def __post_init__(self):
        fields = dataclasses.fields(self)
        fields = {field.name: field for field in fields}
        for name in self.attr2env_var:
            if fields[name].type in (bool, Optional[bool]):
                super().__setattr__(
                    name, parse_boolean_from_env(self.attr2env_var[name])
                )
            elif fields[name].type in (int, Optional[int]):
                super().__setattr__(
                    name, parse_integer_from_env(self.attr2env_var[name])
                )
            else:
                raise ValueError(
                    f"Unsupported type {dataclasses.fields(self)[name].type}"
                )

        super().__setattr__("_initialized", True)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if getattr(self, "_initialized", False) and name in self.attr2env_var:
            fields = dataclasses.fields(self)
            fields = dataclasses.fields(self)
            fields = {field.name: field for field in fields}
            if fields[name].type in (bool, Optional[bool]):
                set_boolean_env_var(self.attr2env_var[name], value)
            elif fields[name].type in (int, Optional[int]):
                set_integer_env_var(self.attr2env_var[name], value)
            else:
                raise ValueError(
                    f"Unsupported type {dataclasses.fields(self)[name].type}"
                )


init_default_env()
oneflow_compiler_config = OneFlowCompilerConfig()
