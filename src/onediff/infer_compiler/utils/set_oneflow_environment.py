import os

def set_oneflow_environment():
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
    os.environ["ONEFLOW_KERNEL_CONV_ENABLE_CUTLASS_IMPL"] = "1"
    os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
    os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
    os.environ["ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT"] = "1"

def _use_graph():
    os.environ["with_graph"] = "1"
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
    os.environ["ONEFLOW_KERNEL_CONV_ENABLE_CUTLASS_IMPL"] = "1"
    os.environ["ONEFLOW_KERNEL_GEMM_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"
    os.environ["ONEFLOW_KERNEL_GEMM_ENABLE_CUTLASS_IMPL"] = "1"
    os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
    os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
    os.environ["ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT"] = "1"
    os.environ["ONEFLOW_KERNEL_GLU_ENABLE_DUAL_GEMM_IMPL"] = "0"
    os.environ["ONEFLOW_MLIR_GROUP_MATMUL_QUANT"] = "1"
    os.environ["ONEFLOW_FUSE_QUANT_TO_MATMUL"] = "0"
    # os.environ["ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH"] = "1"
    # os.environ["ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH"] = "1"

# 使用示例
if __name__ == "__main__":
    set_oneflow_environment()
    print("OneFlow environment variables set successfully.")
