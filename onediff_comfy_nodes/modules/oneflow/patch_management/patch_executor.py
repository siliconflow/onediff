from abc import ABC, abstractmethod
from typing import Dict, List
from comfy.model_patcher import ModelPatcher
from comfy.model_base import BaseModel
from register_comfy.CrossAttentionPatch import CrossAttentionPatch


class PatchExecutorBase(ABC):
    @abstractmethod
    def check_patch(self):
        pass

    @abstractmethod
    def set_patch(self):
        pass

    @abstractmethod
    def get_patch(self):
        pass


class CachedCrossAttentionPatch(PatchExecutorBase):
    def __init__(self) -> None:
        self.patch_name = type(self).__name__

    def check_patch(self, module):
        return hasattr(module, self.patch_name)

    def set_patch(self, module, value: dict):
        setattr(module, self.patch_name, value)

    def get_patch(self, module) -> Dict[str, CrossAttentionPatch]:
        if not self.check_patch(module):
            self.set_patch(module, {})
        return getattr(module, self.patch_name)


class DeepCacheUNetExecutorPatch(PatchExecutorBase):
    def __init__(self) -> None:
        super().__init__()
        self.patch_names = ("deep_cache_unet", "fast_deep_cache_unet")

    def check_patch(self, model_patcher: ModelPatcher):
        return all(hasattr(model_patcher, name) for name in self.patch_names)

    def set_patch(self, model_patcher: ModelPatcher, values):
        assert len(self.patch_names) == len(values)
        for attr, value in zip(self.patch_names, values):
            setattr(model_patcher, attr, value)

    def get_patch(self, model_patcher: ModelPatcher) -> List:
        return [getattr(model_patcher, attr, None) for attr in self.patch_names]

    def copy_to(self, old_model: ModelPatcher, new_model: ModelPatcher):
        values = self.get_patch(old_model)
        self.set_patch(new_model, values)
        new_model.model.use_deep_cache_unet = True

    def is_use_deep_cache_unet(self, module: BaseModel):
        return getattr(module, "use_deep_cache_unet", False)
