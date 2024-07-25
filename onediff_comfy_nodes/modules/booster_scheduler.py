import copy
from functools import singledispatchmethod, wraps
from typing import List

import torch.nn as nn
from comfy import model_management
from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE

from .booster_cache import BoosterCacheService
from .booster_interface import BoosterExecutor, BoosterSettings


def auto_cache_model(func):
    @wraps(func)
    def wrapper(self: "BoosterScheduler", model=None, *args, **kwargs):
        if self.settings is None:
            return func(self, model, *args, **kwargs)
        cached_model_key = self.settings.tmp_cache_key
        cached_model = self.cache_service.get_cached_model(cached_model_key, model)
        if cached_model is not None:
            return cached_model
        cached_model = func(self, model, *args, **kwargs)
        self.cache_service.put(cached_model_key, cached_model)
        return cached_model

    return wrapper


class BoosterScheduler:
    def __init__(
        self,
        booster_executors: List[BoosterExecutor],
        *,
        inplace=True,
        settings: BoosterSettings = None,
    ):
        if not isinstance(booster_executors, (list, tuple)):
            booster_executors = [booster_executors]
        self.booster_executors = booster_executors
        self.inplace = inplace
        self.settings = settings
        self.cache_service = BoosterCacheService()

    def is_empty(self) -> bool:
        """
        Checks if the list of boosters is empty.
        """
        return not self.booster_executors

    @auto_cache_model
    def compile(self, model=None, ckpt_name=None, **kwargs):
        if not self.inplace:
            model = self.copy(model)
        for executor in self.booster_executors:
            model = executor.execute(model, ckpt_name=ckpt_name, **kwargs)

        return model

    def __call__(self, model=None, ckpt_name=None, **kwargs):
        return self.compile(model=model, ckpt_name=ckpt_name, **kwargs)

    @singledispatchmethod
    def copy(self, model):
        raise NotImplementedError(f"Copying {type(model)} is not implemented.")

    @copy.register
    def _(self, model: ModelPatcher):
        model.model = model.model.to("cpu")
        new_modelpatcher = model.clone()
        copied_model: nn.Module = copy.deepcopy(model.model)
        new_modelpatcher.model = copied_model.to(model_management.get_torch_device())
        return new_modelpatcher

    @copy.register
    def _(self, model: VAE):
        new_vae = copy.deepcopy(model)
        return new_vae
