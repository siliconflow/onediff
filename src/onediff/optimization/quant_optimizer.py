import os
import torch.nn as nn
from pathlib import Path
from copy import deepcopy
from ..infer_compiler.utils.version_util import (
    get_support_message,
    is_quantization_enabled,
)

__all__ = ["replace_module_with_quantizable_module"]


def _varify_can_use_quantization():
    if not is_quantization_enabled():
        message = get_support_message()
        print(f"Warning: {message}")
        return False
    return True


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


def modify_sub_module(module, sub_module_name, new_value):
    """Modify a submodule of a module using dot-separated names.

    Args:
        module (nn.Module): The base module.
        sub_module_name (str): Dot-separated name of the submodule.
        new_value: The new value to assign to the submodule.

    """
    parts = sub_module_name.split(".")
    current_module = module

    for i, part in enumerate(parts):
        try:
            if part.isdigit():
                if i == len(parts) - 1:
                    current_module[int(part)] = new_value
                else:
                    current_module = current_module[int(part)]
            else:
                if i == len(parts) - 1:
                    setattr(current_module, part, new_value)
                else:
                    current_module = getattr(current_module, part)
        except (IndexError, AttributeError):
            raise ModuleNotFoundError(f"Submodule {part} not found.")


def replace_module_with_quantizable_module(
    model,  # diffusion_model
    quantize_conv=True,
    quantize_linear=True,
    verbose=False,
    bits=8,
    *,
    inplace=True,
):
    if _varify_can_use_quantization() is False:
        return model

    from diffusers_quant.utils import symm_quantize_sub_module, find_quantizable_modules
    from torch._dynamo import allow_in_graph as maybe_allow_in_graph
    from diffusers_quant.utils import get_quantize_module
    from diffusers_quant import Quantizer

    _use_graph()

    if not inplace:
        model = deepcopy(model)

    def quant(quantizable_modules):
        nonlocal model

        for name, sub_mod in quantizable_modules.items():
            quantizer = Quantizer()
            quantizer.configure(bits=bits, perchannel=True)
            quantizer.find_params(sub_mod.weight.float(), weight=True)
            shape = [-1] + [1] * (len(sub_mod.weight.shape) - 1)
            scale = quantizer.scale.reshape(*shape)
            symm_quantize_sub_module(
                model, name, scale, quantizer.maxq, save_as_float=False
            )

            input_scale = 0
            input_zero_point = 0
            weight_scale = scale.reshape(-1).tolist()
            sub_module_name = name
            sub_calibrate_info = [input_scale, input_zero_point, weight_scale]
            sub_mod = get_quantize_module(
                sub_mod,
                sub_module_name,
                sub_calibrate_info,
                fake_quant=False,
                static=False,
                nbits=8,
                convert_fn=maybe_allow_in_graph,
            )
            modify_sub_module(model, name, sub_mod)

    if quantize_conv:
        conv_modules = find_quantizable_modules(model, module_cls=[nn.Conv2d])
        quant(conv_modules)
        if verbose:
            print(f"{len(conv_modules)=}")
    if quantize_linear:
        linear_modules = find_quantizable_modules(model, module_cls=[nn.Linear])
        quant(linear_modules)
        if verbose:
            print(f"{len(linear_modules)=}")

    green, end = "\033[92m", "\033[0m"
    print(f"{green}Quantized model {type(model)} successfully!{end}")
    return model
