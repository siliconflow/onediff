import importlib
import importlib.metadata
import os

from packaging import version

diffusers_0220_v = version.parse("0.22.0")
diffusers_0240_v = version.parse("0.24.0")
diffusers_0260_v = version.parse("0.26.0")
diffusers_0270_v = version.parse("0.27.0")
diffusers_version = version.parse(importlib.metadata.version("diffusers"))

import diffusers

if diffusers_version < diffusers_0220_v:

    def get_class_obj_and_candidates(
        library_name, class_name, importable_classes, pipelines, is_pipeline_module
    ):
        """Simple helper method to retrieve class object of module as well as potential parent class objects"""
        if is_pipeline_module:
            pipeline_module = getattr(pipelines, library_name)

            class_obj = getattr(pipeline_module, class_name)
            class_candidates = {c: class_obj for c in importable_classes.keys()}
        else:
            if class_name == "UNet2DConditionModel":
                library_name = "onediffx.deep_cache.models.unet_2d_condition"

            if class_name == "UNetSpatioTemporalConditionModel":
                assert (
                    diffusers_version >= diffusers_0240_v
                ), "SVD not support in diffusers-" + str(diffusers_version)
                library_name = (
                    "onediffx.deep_cache.models.unet_spatio_temporal_condition"
                )

            # else we just import it from the library.
            library = importlib.import_module(library_name)

            class_obj = getattr(library, class_name)
            class_candidates = {
                c: getattr(library, c, None) for c in importable_classes.keys()
            }

        return class_obj, class_candidates

else:

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
                component_folder,
                module_file=library_name + ".py",
                class_name=class_name,
            )
            class_candidates = {c: class_obj for c in importable_classes.keys()}
        else:
            # else we just import it from the library.
            if class_name == "UNet2DConditionModel":
                library_name = "onediffx.deep_cache.models.unet_2d_condition"

            if class_name == "UNetSpatioTemporalConditionModel":
                assert (
                    diffusers_version >= diffusers_0240_v
                ), "SVD not support in diffusers-" + str(diffusers_version)
                library_name = (
                    "onediffx.deep_cache.models.unet_spatio_temporal_condition"
                )

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

if diffusers_version >= diffusers_0260_v:
    from diffusers.models.unets import (
        unet_2d_condition as diffusers_unet_2d_condition,
        unet_spatio_temporal_condition as diffusers_unet_spatio_temporal_condition,
    )
else:
    from diffusers.models import unet_2d_condition as diffusers_unet_2d_condition

    if diffusers_version >= diffusers_0240_v:
        from diffusers.models import (
            unet_spatio_temporal_condition as diffusers_unet_spatio_temporal_condition,
        )


def enable_deep_cache_pipeline():
    global ORIGIN_DIFFUDION_GET_CLC_OBJ_CANDIDATES
    global ORIGIN_2D_GET_DOWN_BLOCK
    global ORIGIN_2D_GET_UP_BLOCK
    global ORIGIN_3D_GET_DOWN_BLOCK
    global ORIGIN_3D_GET_UP_BLOCK
    if ORIGIN_DIFFUDION_GET_CLC_OBJ_CANDIDATES is None:
        assert ORIGIN_2D_GET_DOWN_BLOCK is None
        assert ORIGIN_2D_GET_UP_BLOCK is None
        if diffusers_version >= diffusers_0240_v:
            assert ORIGIN_3D_GET_DOWN_BLOCK is None
            assert ORIGIN_3D_GET_UP_BLOCK is None

        if diffusers_version < diffusers_0270_v:
            ORIGIN_DIFFUDION_GET_CLC_OBJ_CANDIDATES = (
                diffusers.pipelines.pipeline_utils.get_class_obj_and_candidates
            )
            diffusers.pipelines.pipeline_utils.get_class_obj_and_candidates = (
                get_class_obj_and_candidates
            )
        else:
            ORIGIN_DIFFUDION_GET_CLC_OBJ_CANDIDATES = (
                diffusers.pipelines.pipeline_loading_utils.get_class_obj_and_candidates
            )
            diffusers.pipelines.pipeline_loading_utils.get_class_obj_and_candidates = (
                get_class_obj_and_candidates
            )

        from .unet_2d_blocks import get_down_block as get_2d_down_block

        ORIGIN_2D_GET_DOWN_BLOCK = diffusers_unet_2d_condition.get_down_block
        diffusers_unet_2d_condition.get_down_block = get_2d_down_block

        from .unet_2d_blocks import get_up_block as get_2d_up_block

        ORIGIN_2D_GET_UP_BLOCK = diffusers_unet_2d_condition.get_up_block
        diffusers_unet_2d_condition.get_up_block = get_2d_up_block

        if diffusers_version >= diffusers_0240_v:
            from .unet_3d_blocks import get_down_block as get_3d_down_block

            ORIGIN_3D_GET_DOWN_BLOCK = (
                diffusers_unet_spatio_temporal_condition.get_down_block
            )
            diffusers_unet_spatio_temporal_condition.get_down_block = get_3d_down_block

            from .unet_3d_blocks import get_up_block as get_3d_up_block

            ORIGIN_3D_GET_UP_BLOCK = (
                diffusers_unet_spatio_temporal_condition.get_up_block
            )
            diffusers_unet_spatio_temporal_condition.get_up_block = get_3d_up_block


def disable_deep_cache_pipeline():
    global ORIGIN_DIFFUDION_GET_CLC_OBJ_CANDIDATES
    global ORIGIN_2D_GET_DOWN_BLOCK
    global ORIGIN_2D_GET_UP_BLOCK
    global ORIGIN_3D_GET_DOWN_BLOCK
    global ORIGIN_3D_GET_UP_BLOCK
    if ORIGIN_DIFFUDION_GET_CLC_OBJ_CANDIDATES is None:
        assert ORIGIN_2D_GET_DOWN_BLOCK is None
        assert ORIGIN_2D_GET_UP_BLOCK is None
        if diffusers_version >= diffusers_0240_v:
            assert ORIGIN_3D_GET_DOWN_BLOCK is None
            assert ORIGIN_3D_GET_UP_BLOCK is None
        return
    if diffusers_version < diffusers_0270_v:
        diffusers.pipelines.pipeline_utils.get_class_obj_and_candidates = (
            ORIGIN_DIFFUDION_GET_CLC_OBJ_CANDIDATES
        )
    else:
        diffusers.pipelines.pipeline_loading_utils.get_class_obj_and_candidates = (
            ORIGIN_DIFFUDION_GET_CLC_OBJ_CANDIDATES
        )
    diffusers_unet_2d_condition.get_down_block = ORIGIN_2D_GET_DOWN_BLOCK
    diffusers_unet_2d_condition.get_up_block = ORIGIN_2D_GET_UP_BLOCK
    if diffusers_version >= diffusers_0240_v:
        diffusers_unet_spatio_temporal_condition.get_down_block = (
            ORIGIN_3D_GET_DOWN_BLOCK
        )
        diffusers_unet_spatio_temporal_condition.get_up_block = ORIGIN_3D_GET_UP_BLOCK


__all__ = [
    "enable_deep_cache_pipeline",
    "disable_deep_cache_pipeline",
]
