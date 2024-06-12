import torch
import traceback
from collections import OrderedDict
from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE
from onediff.torch_utils.module_operations import get_sub_module
from onediff.utils.import_utils import is_oneflow_available

if is_oneflow_available():
    from .oneflow.utils.booster_utils import is_using_oneflow_backend


def switch_to_cached_model(new_model: ModelPatcher, cache_model):
    assert type(new_model.model) == type(cache_model)
    for k, v in new_model.model.state_dict().items():
        cached_v: torch.Tensor = get_sub_module(cache_model, k)
        assert v.dtype == cached_v.dtype
        cached_v.copy_(v)
    new_model.model = cache_model
    return new_model


class BoosterCacheService:
    _cache = OrderedDict()

    def put(self, key, model):
        if key is None:
            return
        # oneflow backends output image error
        if is_oneflow_available() and is_using_oneflow_backend(model):
            return
        self._cache[key] = model.model

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
