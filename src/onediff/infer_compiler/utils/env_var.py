import dataclasses
import os
from typing import Optional


def parse_boolean_from_env(env_var, default_value=None):
    env_var = os.getenv(env_var)
    if env_var is None:
        return default_value
    env_var = env_var.lower()
    return env_var in ("1", "true", "yes", "on", "y")


def set_boolean_env_var(env_var: str, val: Optional[bool]):
    if val is None:
        os.environ.pop(env_var, None)
    else:
        os.environ[env_var] = "1" if val else "0"


def parse_integer_from_env(env_var, default_value=None):
    env_var = os.getenv(env_var)
    if env_var is None:
        return default_value
    return int(env_var)


def set_integer_env_var(env_var: str, val: Optional[int]):
    if val is None:
        os.environ.pop(env_var, None)
    else:
        os.environ[env_var] = str(int(val))


def _set_env_vars(field2env_var, options):
    for field in dataclasses.fields(options):
        field_name = field.name
        field_value = getattr(options, field_name)
        if field_value is None or field_name not in field2env_var:
            continue
        env_var = field2env_var[field_name]
        set_env_var = None
        if field.type in (bool, Optional[bool]):
            set_env_var = set_boolean_env_var
        elif field.type in (int, Optional[int]):
            set_env_var = set_integer_env_var
        else:
            raise ValueError(f"Unsupported type {field.type}")
        set_env_var(env_var, field_value)


def set_oneflow_env_vars(options):
    field2env_var = {
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
        "kernel_enable_conv2d_tuning_warmup": "ONEFLOW_CONV2D_KERNEL_ENABLE_TUNING_WARMUP",
        "kernel_gemm_enable_cutlass_impl": "ONEFLOW_KERNEL_GEMM_ENABLE_CUTLASS_IMPL",
        "kernel_glu_enable_dual_gemm_impl": "ONEFLOW_KERNEL_GLU_ENABLE_DUAL_GEMM_IMPL",
        "kernel_glu_enable_y_gemm_impl": "ONEFLOW_KERNEL_GLU_ENABLE_Y_GEMM_IMPL",
        "kernel_glu_quant_enable_dual_gemm_impl": "ONEFLOW_KERNEL_GLU_QUANT_ENABLE_DUAL_GEMM_IMPL",
        "conv_allow_half_precision_accumulation": "ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION",
        "matmul_allow_half_precision_accumulation": "ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION",
        "attention_allow_half_precision_accumulation": "ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_ACCUMULATION",
        "attention_allow_half_precision_score_accumulation_max_m": "ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_SCORE_ACCUMULATION_MAX_M",
    }
    _set_env_vars(field2env_var, options)


def set_oneflow_default_env_vars():
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


def set_nexfort_env_vars(options):
    field2env_var = {}
    _set_env_vars(field2env_var, options)


def set_nexfort_default_env_vars():
    pass


def set_default_env_vars():
    set_oneflow_default_env_vars()
    set_nexfort_default_env_vars()
