import torch 
from functools import partial, singledispatchmethod
from typing import Optional

from comfy.controlnet import ControlLora, ControlNet
from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE

# from onediff.infer_compiler import compile
from nexfort.compilers import nexfort_compile

from ..booster_interface import BoosterExecutor


class BasicNexFortBoosterExecutor(BoosterExecutor):
    # https://pytorch.org/docs/stable/_modules/torch.html#compile
    def __init__(
        self,
    ):
        super().__init__()
        # self.compile_fn = partial(compile, backend="nexfort")
        self.compile_fn = partial(nexfort_compile)

    
    @singledispatchmethod
    def execute(self, model, ckpt_name=None, **kwargs):
        raise NotImplementedError(f"Cannot execute {type(model)=}")

    @execute.register(ModelPatcher)
    @torch.no_grad()
    def _(self, model, ckpt_name: Optional[str] = None, **kwargs):
        model.model.diffusion_model = self.compile_fn(model.model.diffusion_model)
        return model
    
    @execute.register(VAE)
    @torch.no_grad()
    def _(self, model, ckpt_name: Optional[str] = None, **kwargs):
        # model.first_stage_model = torch.compile(model.first_stage_model, **self.compile_kwargs)
        model.first_stage_model = self.compile_fn(model.first_stage_model)
        return model

    @execute.register(ControlNet)
    @torch.no_grad()
    def _(self, model, ckpt_name: Optional[str] = None, **kwargs):
        torch_model = model.control_model
        compiled_model = self.compile_fn(torch_model)
        model.control_model = compiled_model
        return model
