import torch
from comfy import model_management
from comfy.model_base import SVD_img2vid
from comfy.model_patcher import ModelPatcher
from comfy.model_base import BaseModel

from onediff.infer_compiler.oneflow import \
    OneflowDeployableModule as DeployableModule
from onediff.infer_compiler.utils import set_boolean_env_var

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
    if isinstance(module, ModelPatcher):
        deep_cache_module = getattr(module,"deep_cache_unet", None)
        if deep_cache_module and  isinstance(deep_cache_module, DeployableModule):
            return True
        diff_model = module.model.diffusion_model
        return isinstance(diff_model, DeployableModule)
    
    if isinstance(module, BaseModel):
        if getattr(module, "use_deep_cache_unet", False):
            return True
        return isinstance(module.diffusion_model, DeployableModule)

    if isinstance(module, DeployableModule):
        return True
    
    raise RuntimeError("")
        




