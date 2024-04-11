import copy
from functools import singledispatchmethod
from typing import List
from comfy.model_patcher import ModelPatcher

from .booster_interface import BoosterExecutor


class BoosterScheduler:
    def __init__(self, booster_executors: List[BoosterExecutor], * , inplace = True):
        if not isinstance(booster_executors, (list, tuple)):
            booster_executors = [booster_executors]
        self.booster_executors = booster_executors
        self.inplace = inplace


    def is_empty(self) -> bool:
        """
        Checks if the list of boosters is empty.
        """
        return not self.booster_executors
    
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
        new_modelpatcher = model.clone()
        new_modelpatcher.model = copy.deepcopy(model.model)
        return new_modelpatcher

    

    
    
