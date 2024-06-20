import torch
import traceback
from collections import OrderedDict
from comfy.model_patcher import ModelPatcher
from functools import singledispatch
from comfy.sd import VAE
from onediff.torch_utils.module_operations import get_sub_module


@singledispatch
def switch_to_cached_model(new_model, cached_model):
    raise NotImplementedError(type(new_model))


@switch_to_cached_model.register
def _(new_model: ModelPatcher, cached_model):
    assert type(new_model.model) == type(
        cached_model
    ), f"Model type mismatch: expected {type(cached_model)}, got {type(new_model.model)}"
    for k, v in new_model.model.state_dict().items():
        cached_v: torch.Tensor = get_sub_module(cached_model, k)
        assert v.dtype == cached_v.dtype
        cached_v.copy_(v)
    new_model.model = cached_model
    return new_model


@switch_to_cached_model.register
def _(new_model: VAE, cached_model):
    assert type(new_model.first_stage_model) == type(cached_model)
    for k, v in new_model.model.state_dict().items():
        cached_v: torch.Tensor = get_sub_module(cached_model, k)
        assert v.dtype == cached_v.dtype
        cached_v.copy_(v)
    new_model.first_stage_model = cached_model
    return new_model


@singledispatch
def get_cached_model(model):
    return None
    # raise NotImplementedError(type(model))


@get_cached_model.register
def _(model: ModelPatcher):
    return model.model


@get_cached_model.register
def _(model: VAE):
    # TODO(TEST) if support cache
    return None
    # return model.first_stage_model


class BoosterCacheService:
    _cache = OrderedDict()

    def put(self, key, model):
        if key is None:
            return
        # oneflow backends output image error
        cached_model = get_cached_model(model)
        if cached_model:
            self._cache[key] = cached_model

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
