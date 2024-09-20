from typing import Union

import oneflow

import torch
from comfy import model_management
from comfy.model_base import BaseModel, SVD_img2vid
from comfy.model_patcher import ModelPatcher

from onediff.infer_compiler.backends.oneflow import (
    OneflowDeployableModule as DeployableModule,
)
from onediff.utils import set_boolean_env_var
from onediff.utils.import_utils import is_oneflow_available

from ..patch_management import create_patch_executor, PatchType


def set_compiled_options(module: DeployableModule, graph_file="unet"):
    assert isinstance(module, DeployableModule)
    compile_options = module._deployable_module_options
    compile_options.graph_file = graph_file
    compile_options.graph_file_device = model_management.get_torch_device()


def is_fp16_model(model):
    """
    Check if the model is using FP16 (Half Precision).

    Parameters:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        bool: True if the model is using FP16, False otherwise.
    """
    for param in model.parameters():
        if param.dtype == torch.float16:
            return True
    return False


def get_model_type(model):
    """
    Get the types of the parameters in the model.

    Parameters:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        set: A set containing the types of the parameters in the model.
    """
    type_set = set()
    for param in model.parameters():
        type_set.add(param.dtype)
    return type_set


def set_environment_for_svd_img2vid(model: ModelPatcher):
    if isinstance(model, ModelPatcher) and isinstance(model.model, SVD_img2vid):
        # TODO(fengwen) check it affect performance
        # To avoid overflow issues while maintaining performance,
        # refer to: https://github.com/siliconflow/onediff/blob/09a94df1c1a9c93ec8681e79d24bcb39ff6f227b/examples/image_to_video.py#L112
        set_boolean_env_var(
            "ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_SCORE_ACCUMULATION_MAX_M", False
        )


def is_using_oneflow_backend(module):
    # First, check if oneflow is available and CUDA is enabled
    if is_oneflow_available() and not oneflow.cuda.is_available():
        print("OneFlow CUDA support is not available")
        return False

    # Check if the module
    if isinstance(module, oneflow.nn.Module):
        return True

    dc_patch_executor = create_patch_executor(PatchType.DCUNetExecutorPatch)
    if isinstance(module, ModelPatcher):
        deep_cache_module = dc_patch_executor.get_patch(module)
        if deep_cache_module[0] and isinstance(deep_cache_module[0], DeployableModule):
            return True
        if hasattr(module.model, "diffusion_model"):
            diff_model = module.model.diffusion_model
            return isinstance(diff_model, DeployableModule)
        else:
            return False

    if isinstance(module, BaseModel):
        if dc_patch_executor.is_use_deep_cache_unet(module):
            return True
        if hasattr(module, "diffusion_model"):
            return isinstance(module.diffusion_model, DeployableModule)
        else:
            return False

    if isinstance(module, DeployableModule):
        return True

    if hasattr(module, "parameters"):
        for param in module.parameters():
            if isinstance(param, oneflow.Tensor):
                return True

    warn_msg = (
        f"OneFlow backend is not detected for the module, the module is {type(module)}"
    )
    print(warn_msg)
    # If none of the above conditions are met, it's not using OneFlow backend
    return False


def clear_deployable_module_cache_and_unbind(
    module: Union[ModelPatcher, DeployableModule]
):
    if isinstance(module, ModelPatcher):
        dcu_patch = create_patch_executor(PatchType.DCUNetExecutorPatch)
        if dcu_patch.check_patch(module):
            for sub_module in dcu_patch.get_patch(module):
                sub_module._clear_old_graph()

        diff_model = module.model.diffusion_model
        if isinstance(diff_model, DeployableModule):
            diff_model._clear_old_graph()
        create_patch_executor(PatchType.CachedCrossAttentionPatch).clear_patch(
            diff_model
        )
        create_patch_executor(PatchType.UNetExtraInputOptions).clear_patch(diff_model)
    elif isinstance(module, DeployableModule):
        diff_model = module
        diff_model._clear_old_graph()
        create_patch_executor(PatchType.CachedCrossAttentionPatch).clear_patch(
            diff_model
        )
        create_patch_executor(PatchType.CrossAttentionForwardMasksPatch).clear_patch(
            diff_model
        )
    else:
        raise RuntimeError(f"Unexpected module type: {type(module)}.")
