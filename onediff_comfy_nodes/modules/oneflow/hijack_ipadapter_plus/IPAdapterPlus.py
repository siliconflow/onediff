"""hijack ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py

hijack ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py"""

import functools

from onediff.utils.log_utils import logger

from ..utils.booster_utils import is_using_oneflow_backend
from ._config import ipadapter_plus_hijacker, ipadapter_plus_pt
from .set_model_patch_replace import set_model_patch_replace_v2

set_model_patch_replace_fn_pt = ipadapter_plus_pt.IPAdapterPlus.set_model_patch_replace
ipadapter_execute = ipadapter_plus_pt.IPAdapterPlus.ipadapter_execute


def cache_init(original_init):
    @functools.wraps(original_init)
    def new_init(self, ipadapter_model, *args, **kwargs):
        cache_key = args + tuple(kwargs.values())
        # import time
        # t0 = time.time()
        if cache_key in ipadapter_model:
            cached_instance = ipadapter_model[cache_key]
            self.__dict__ = cached_instance.__dict__
            logger.debug("Using cached IPAdapter instance")
        else:
            original_init(self, ipadapter_model, *args, **kwargs)
            ipadapter_model[cache_key] = self
            logger.debug("Caching new IPAdapter instance")

        # print(f'dur: {time.time() - t0}')

    return new_init


def cond_func(org_fn, model, *args, **kwargs):
    return is_using_oneflow_backend(model)


def ipadapter_execute_of(org_fn, model, *args, **kwargs):
    IPAdapter = ipadapter_plus_pt.IPAdapterPlus.IPAdapter
    _original_init = IPAdapter.__init__
    IPAdapter.__init__ = cache_init(_original_init)
    output = org_fn(model, *args, **kwargs)
    IPAdapter.__init__ = _original_init
    return output


ipadapter_plus_hijacker.register(
    set_model_patch_replace_fn_pt, set_model_patch_replace_v2, cond_func
)
ipadapter_plus_hijacker.register(ipadapter_execute, ipadapter_execute_of, cond_func)
