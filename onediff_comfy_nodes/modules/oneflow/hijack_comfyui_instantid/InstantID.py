import functools

from onediff.utils.log_utils import logger

from ..hijack_ipadapter_plus.set_model_patch_replace import set_model_patch_replace_v2

from ..utils.booster_utils import is_using_oneflow_backend
from ._config import comfyui_instantid_hijacker, comfyui_instantid_pt

set_model_patch_replace_fn_pt = comfyui_instantid_pt.InstantID._set_model_patch_replace
apply_instantid = comfyui_instantid_pt.InstantID.ApplyInstantID.apply_instantid
InstantID = comfyui_instantid_pt.InstantID.InstantID
patch_attention = comfyui_instantid_pt.InstantID.InstantIDAttentionPatch.patch_attention


def cache_init(original_init):
    @functools.wraps(original_init)
    def new_init(self, instantid_model, *args, **kwargs):
        cache_key = args + tuple(kwargs.values())
        # import time
        # t0 = time.time()
        if cache_key in instantid_model:
            cached_instance = instantid_model[cache_key]
            self.__dict__ = cached_instance.__dict__
            logger.debug("Using cached InstantID instance")
        else:
            original_init(self, instantid_model, *args, **kwargs)
            instantid_model[cache_key] = self
            logger.debug("Caching new InstantID instance")

        # print(f'dur: {time.time() - t0}')

    return new_init


def cond_func(org_fn, model, *args, **kwargs):
    return is_using_oneflow_backend(model)


def apply_instantid_of(org_fn, self, *args, **kwargs):
    model = kwargs["model"]
    _org_init = InstantID.__init__
    InstantID.__init__ = cache_init(_org_init)
    output = org_fn(self, *args, **kwargs)
    InstantID.__init__ = _org_init
    return output


def apply_instantid_cond(org_fn, self, *args, **kwargs):
    return is_using_oneflow_backend(kwargs["model"])


comfyui_instantid_hijacker.register(
    set_model_patch_replace_fn_pt, set_model_patch_replace_v2, cond_func
)

comfyui_instantid_hijacker.register(
    apply_instantid, apply_instantid_of, apply_instantid_cond
)

comfyui_instantid_hijacker.register(
    patch_attention, apply_instantid_of, apply_instantid_cond
)
