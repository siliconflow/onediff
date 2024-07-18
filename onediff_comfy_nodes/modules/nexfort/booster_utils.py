from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher
from onediff.infer_compiler.backends.nexfort.deployable_module import (
    NexfortDeployableModule as DeployableModule,
)


def clear_deployable_module_cache_and_unbind(*args, **kwargs):
    raise RuntimeError(f"TODO")


def is_using_nexfort_backend(module):
    if isinstance(module, ModelPatcher):
        if hasattr(module.model, "diffusion_model"):
            diff_model = module.model.diffusion_model
            return isinstance(diff_model, DeployableModule)
    if isinstance(module, BaseModel):
        if hasattr(module, "diffusion_model"):
            return isinstance(module.diffusion_model, DeployableModule)
    return False
