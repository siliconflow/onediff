from enum import Enum

from .patch_executor import (
    CachedCrossAttentionPatch,
    DeepCacheUNetExecutorPatch,
    UiNodeWithIndexPatch,
    UNetExtraInputOptions,
)


class PatchType(Enum):
    CachedCrossAttentionPatch = CachedCrossAttentionPatch
    UNetExtraInputOptions = UNetExtraInputOptions
    DCUNetExecutorPatch = DeepCacheUNetExecutorPatch
    UiNodeWithIndexPatch = UiNodeWithIndexPatch


def create_patch_executor(selected_patch_type):
    for patch_type in PatchType:
        if selected_patch_type == patch_type:
            return patch_type.value()
    raise NotImplementedError(selected_patch_type)


if __name__ == "__main__":
    patch_executor = create_patch_executor(PatchType.CachedCrossAttentionPatch)
    print(patch_executor)
