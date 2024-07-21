from abc import ABC, abstractmethod
from typing import Dict, List

from comfy.model_base import BaseModel

from comfy.model_patcher import ModelPatcher


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


class UiNodeWithIndexPatch(PatchExecutorBase):
    DEFAULT_VALUE = -1
    INCREMENT_VALUE = 1

    def __init__(self) -> None:
        self.patch_name = type(self).__name__

    def check_patch(self, module: ModelPatcher) -> bool:
        return hasattr(module, self.patch_name)

    def set_patch(self, module: ModelPatcher, value: int):
        setattr(module, self.patch_name, value)

    def get_patch(self, module: ModelPatcher) -> int:
        return getattr(module, self.patch_name, self.DEFAULT_VALUE)

    def copy_to(self, old_model: ModelPatcher, new_model: ModelPatcher):
        value = self.get_patch(old_model)
        self.set_patch(new_model, value + self.INCREMENT_VALUE)


class CachedCrossAttentionPatch(PatchExecutorBase):
    def __init__(self) -> None:
        self.patch_name = type(self).__name__

    def check_patch(self, module):
        return hasattr(module, self.patch_name)

    def set_patch(self, module, value: dict):
        setattr(module, self.patch_name, value)

    def get_patch(self, module) -> Dict[str, any]:
        if not self.check_patch(module):
            self.set_patch(module, {})
        return getattr(module, self.patch_name)

    def clear_patch(self, module):
        if self.check_patch(module):
            self.get_patch(module).clear()


class CrossAttentionForwardMasksPatch(PatchExecutorBase):
    def __init__(self) -> None:
        """Will be abandoned"""
        self.patch_name = "forward_masks"

    def check_patch(self, module):
        return hasattr(module, self.patch_name)

    def set_patch(self, module, value):
        raise NotImplementedError()

    def get_patch(self, module) -> Dict:
        if not self.check_patch(module):
            setattr(module, self.patch_name, {})
        return getattr(module, self.patch_name)

    def clear_patch(self, module):
        if self.check_patch(module):
            self.get_patch(module).clear()


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


class UNetExtraInputOptions(PatchExecutorBase):
    def __init__(self) -> None:
        """UNetExtraInputOptions"""
        super().__init__()
        self.patch_name = type(self).__name__

    def check_patch(self, module):
        return hasattr(module, self.patch_name)

    def set_patch(self, module, value: Dict):
        """
        Bind extra input options to the specified module.
        For UNet extra input options, the value is a dictionary.

        Args:
            module: The module object to set the patch attribute on.
            value (Dict): The extra input options to bind to the module.
        """
        setattr(module, self.patch_name, value)

    def get_patch(self, module) -> Dict:
        if not self.check_patch(module):
            self.set_patch(module, {})
        return getattr(module, self.patch_name)

    def clear_patch(self, module):
        if self.check_patch(module):
            self.get_patch(module).clear()
