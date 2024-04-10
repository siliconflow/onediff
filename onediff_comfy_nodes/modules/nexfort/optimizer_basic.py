from functools import partial, singledispatchmethod
from typing import Optional

from comfy.controlnet import ControlLora, ControlNet
from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE
from onediff.infer_compiler import (CompileOptions, NexfortCompileOptions,
                                    compile)

from ..optimizer_interface import OptimizerExecutor


class NexFortOptimizerExecutor(OptimizerExecutor):
    # https://pytorch.org/docs/stable/_modules/torch.html#compile
    def __init__(self, fullgraph=False, dynamic=None, backend='inductor', mode='default', options=None, disable=False):
        super().__init__()
        self.compile_kwargs = {
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "backend": backend,
            "options": options,
            "mode": mode,
            "disable": disable,
        }
        compiled_options = CompileOptions()
        compiled_options.nexfort = NexfortCompileOptions(**self.compile_kwargs)
        self.compile_fn = partial(compile, options=compiled_options)


    @singledispatchmethod
    def execute(self, model, ckpt_name=None, **kwargs):
        raise NotImplementedError(f'Cannot execute {type(model)=}')
    

    @execute.register(VAE)
    def _(self, model, ckpt_name: Optional[str] = None, **kwargs):
        # model.first_stage_model = torch.compile(model.first_stage_model, **self.compile_kwargs)
        model.first_stage_model = self.compile_fn(model.first_stage_model)
        return model
    
    @execute.register(ControlNet)
    def _(self, model, ckpt_name: Optional[str] = None, **kwargs):
        torch_model = model.control_model
        compiled_model = self.compile_fn(torch_model)
        model.control_model = compiled_model
        return model
    