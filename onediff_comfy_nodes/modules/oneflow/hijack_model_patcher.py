from comfy.model_patcher import ModelPatcher
from onediff.utils import logger

from ..sd_hijack_utils import Hijacker
from .patch_management import create_patch_executor, PatchType
from .utils.booster_utils import is_using_oneflow_backend


def clone_oneflow(org_fn, self, *args, **kwargs):
    n = org_fn(self, *args, **kwargs)
    create_patch_executor(PatchType.UiNodeWithIndexPatch).copy_to(self, n)
    dc_patch_executor = create_patch_executor(PatchType.DCUNetExecutorPatch)
    if dc_patch_executor.check_patch(self):
        dc_patch_executor.copy_to(self, n)
    return n


def cond_func(org_fn, self):
    return is_using_oneflow_backend(self)


def unpatch_model_oneflow(org_fn, self, device_to=None, unpatch_weights=True):
    if unpatch_weights:
        logger.warning(
            f"{type(self.model.diffusion_model)} is quantized by onediff, so unpatching is skipped."
        )
    return


def unpatch_model_cond_func(org_fn, self, *args, **kwargs):
    if hasattr(self.model, "diffusion_model") and hasattr(
        self.model.diffusion_model, "_deployable_module_quantized"
    ):
        return self.model.diffusion_model._deployable_module_quantized
    return False


model_patch_hijacker = Hijacker()

model_patch_hijacker.register(ModelPatcher.clone, clone_oneflow, cond_func)
model_patch_hijacker.register(
    ModelPatcher.unpatch_model, unpatch_model_oneflow, unpatch_model_cond_func
)
