import copy
from functools import singledispatchmethod
from typing import List

from comfy.model_patcher import ModelPatcher
from onediff.infer_compiler.oneflow import \
    OneflowDeployableModule as DeployableModule

from .optimizer_interface import OptimizerExecutor


class OptimizerScheduler:
    def __init__(self, optimizers: List[OptimizerExecutor], * , inplace = True):
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        self.optimizers = optimizers
        self.inplace = inplace


    def is_empty(self) -> bool:
        """
        Checks if the list of optimizers is empty.
        """
        return not self.optimizers
    
    def compile(self, model=None, ckpt_name=None, **kwargs):
        if not self.inplace:
            model = self.copy(model)
        for optimizer in self.optimizers:
            
            model = optimizer.execute(model, ckpt_name=ckpt_name, **kwargs)
        return model

    def __call__(self, model=None, ckpt_name=None, **kwargs):
        return self.compile(model=model, ckpt_name=ckpt_name, **kwargs)
    
    @singledispatchmethod
    def copy(self, model):
        raise NotImplementedError(f"Copying {type(model)} is not implemented.")
    
    @copy.register
    def _(self, model: ModelPatcher):
        assert not isinstance(model.model.diffusion_model, DeployableModule)
        new_modelpatcher = model.clone()
        new_modelpatcher.model = copy.deepcopy(model.model)
        return new_modelpatcher

    

    
    
