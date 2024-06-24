import torch
import traceback
from collections import OrderedDict
from functools import singledispatch
from comfy.controlnet import ControlLora, ControlNet
from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE
from onediff.torch_utils.module_operations import get_sub_module
from onediff.utils.import_utils import is_oneflow_available
from .._config import is_disable_oneflow_backend


if not is_disable_oneflow_backend() and is_oneflow_available():
    from .oneflow.utils.booster_utils import is_using_oneflow_backend


@singledispatch
def get_target_model(model):
    raise NotImplementedError(f"{type(model)=} cache is not supported.")

@get_target_model.register(ModelPatcher)
def _(model):
    return model.model

@get_target_model.register(VAE)
def _(model):
    return model.first_stage_model

@get_target_model.register(ControlNet)
def _(model):
    return model.control_model

@get_target_model.register(ControlLora)
def _(model):
    return model


def switch_to_cached_model(new_model, cache_model):
    target_model = get_target_model(new_model)
    assert type(target_model) == type(cache_model)
    for k, v in target_model.state_dict().items():
        cached_v: torch.Tensor = get_sub_module(cache_model, k)
        assert v.dtype == cached_v.dtype
        cached_v.copy_(v)
    if isinstance(new_model, ModelPatcher):
        new_model.model = cache_model
    elif isinstance(new_model, VAE):
        new_model.first_stage_model = cache_model
    elif isinstance(new_model, ControlNet):
        new_model.control_model = cache_model
    elif isinstance(new_model, ControlLora):
        new_model = cache_model
    else:
        raise NotImplementedError(f"{type(new_model)=} cache is not supported.")
    return new_model


class BoosterCacheService:
    _cache = OrderedDict()

    def put(self, key, model):
        if key is None:
            return
        # oneflow backends output image error
        if not is_disable_oneflow_backend() and is_oneflow_available() and is_using_oneflow_backend(model):
            return
        self._cache[key] = get_target_model(model)

    def get(self, key, default=None):
        return self._cache.get(key, default)

    def get_cached_model(self, key, model):
        cached_model = self.get(key, None)
        print(f"Cache lookup: Key='{key}', Cached Model Type='{type(cached_model)}'")
        if cached_model is not None:
            try:
                return switch_to_cached_model(model, cached_model)
            except Exception as e:
                print("An exception occurred when switching to cached model:")
                print(traceback.format_exc())
                del self._cache[key]
                torch.cuda.empty_cache()

        return None
