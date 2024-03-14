import os
import diffusers

import importlib

def get_class_obj_and_candidates(
    library_name,
    class_name,
    importable_classes,
    pipelines,
    is_pipeline_module,
    component_name=None,
    cache_dir=None,
):
    """Simple helper method to retrieve class object of module as well as potential parent class objects"""
    component_folder = os.path.join(cache_dir, component_name)

    if is_pipeline_module:
        pipeline_module = getattr(pipelines, library_name)

        class_obj = getattr(pipeline_module, class_name)
        class_candidates = {c: class_obj for c in importable_classes.keys()}
    elif os.path.isfile(os.path.join(component_folder, library_name + ".py")):
        # load custom component
        class_obj = get_class_from_dynamic_module(
            component_folder, module_file=library_name + ".py", class_name=class_name
        )
        class_candidates = {c: class_obj for c in importable_classes.keys()}
    else:
        # else we just import it from the library.
        if class_name == "UNet2DConditionModel":
            library_name = "onediffx.deep_cache.models.unet_2d_condition"

        if class_name == "UNetSpatioTemporalConditionModel":
            assert diffusers_version >= version.parse("0.24.0"), (
                "SVD not support in diffusers-" + diffusers_version
            )
            library_name = "onediffx.deep_cache.models.unet_spatio_temporal_condition"

        library = importlib.import_module(library_name)
        class_obj = getattr(library, class_name)
        class_candidates = {
            c: getattr(library, c, None) for c in importable_classes.keys()
        }

    return class_obj, class_candidates


ORIGIN_DIFFUDION_PIPELINE = None
ORIGIN_DIFFUDION_GET_CLC_OBJ_CANDIDATES = None
ORIGIN_2D_GET_DOWN_BLOCK = None
ORIGIN_2D_GET_UP_BLOCK = None

ORIGIN_3D_GET_DOWN_BLOCK = None
ORIGIN_3D_GET_UP_BLOCK = None


def enable_deep_cache_pipeline():
    global ORIGIN_DIFFUDION_GET_CLC_OBJ_CANDIDATES
    global ORIGIN_2D_GET_DOWN_BLOCK
    global ORIGIN_2D_GET_UP_BLOCK
    global ORIGIN_3D_GET_DOWN_BLOCK
    global ORIGIN_3D_GET_UP_BLOCK
    if ORIGIN_DIFFUDION_GET_CLC_OBJ_CANDIDATES is None:
        assert ORIGIN_2D_GET_DOWN_BLOCK is None
        assert ORIGIN_2D_GET_UP_BLOCK is None
        assert ORIGIN_3D_GET_DOWN_BLOCK is None
        assert ORIGIN_3D_GET_UP_BLOCK is None
        ORIGIN_DIFFUDION_GET_CLC_OBJ_CANDIDATES = diffusers.pipelines.pipeline_utils.get_class_obj_and_candidates
        diffusers.pipelines.pipeline_utils.get_class_obj_and_candidates = get_class_obj_and_candidates

        from .unet_2d_blocks import get_down_block as get_2d_down_block

        ORIGIN_2D_GET_DOWN_BLOCK = diffusers.models.unet_2d_condition.get_down_block
        diffusers.models.unet_2d_condition.get_down_block = get_2d_down_block

        from .unet_2d_blocks import get_up_block as get_2d_up_block

        ORIGIN_2D_GET_UP_BLOCK = diffusers.models.unet_2d_condition.get_up_block
        diffusers.models.unet_2d_condition.get_up_block = get_2d_up_block

        from .unet_3d_blocks import get_down_block as get_3d_down_block

        ORIGIN_3D_GET_DOWN_BLOCK = (
            diffusers.models.unet_spatio_temporal_condition.get_down_block
        )
        diffusers.models.unet_spatio_temporal_condition.get_down_block = (
            get_3d_down_block
        )

        from .unet_3d_blocks import get_up_block as get_3d_up_block

        ORIGIN_3D_GET_UP_BLOCK = (
            diffusers.models.unet_spatio_temporal_condition.get_up_block
        )
        diffusers.models.unet_spatio_temporal_condition.get_up_block = get_3d_up_block


def disable_deep_cache_pipeline():
    global ORIGIN_DIFFUDION_GET_CLC_OBJ_CANDIDATES
    global ORIGIN_2D_GET_DOWN_BLOCK
    global ORIGIN_2D_GET_UP_BLOCK
    global ORIGIN_3D_GET_DOWN_BLOCK
    global ORIGIN_3D_GET_UP_BLOCK
    if ORIGIN_DIFFUDION_GET_CLC_OBJ_CANDIDATES is None:
        assert ORIGIN_2D_GET_DOWN_BLOCK is None
        assert ORIGIN_2D_GET_UP_BLOCK is None
        assert ORIGIN_3D_GET_DOWN_BLOCK is None
        assert ORIGIN_3D_GET_UP_BLOCK is None
        return
    diffusers.pipelines.pipeline_utils.get_class_obj_and_candidates = ORIGIN_DIFFUDION_GET_CLC_OBJ_CANDIDATES
    diffusers.models.unet_2d_condition.get_down_block = ORIGIN_2D_GET_DOWN_BLOCK
    diffusers.models.unet_2d_condition.get_up_block = ORIGIN_2D_GET_UP_BLOCK


__all__ = [
    "enable_deep_cache_pipeline",
    "disable_deep_cache_pipeline",
]
