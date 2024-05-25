import torch
from functools import partial, singledispatchmethod
from typing import Optional

from comfy.controlnet import ControlLora, ControlNet
from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE

from onediff.infer_compiler import compile
from nexfort.utils.memory_format import apply_memory_format
from .onediff_controlnet import OneDiffControlLora
from ..booster_interface import BoosterExecutor


def state_dict_hook(module, state_dict, prefix, local_metadata):
    new_state_dict = type(state_dict)()
    for k, v in state_dict.items():
        # diffusion_model._orig_mod.out.2.weight => diffusion_model.out.2.weight
        if k.startswith("diffusion_model._orig_mod"):
            x = k.split(".")
            new_k = ".".join(x[:1] + x[2:])
            new_state_dict[new_k] = v
        elif k.startswith("_orig_mod"):
            x = k.split(".")
            new_k = ".".join(x[1:])
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


class BasicNexFortBoosterExecutor(BoosterExecutor):
    # https://pytorch.org/docs/stable/_modules/torch.html#compile
    def __init__(
        self,
        mode: str = "max-optimize:max-autotune:freezing:benchmark:cudagraphs",
        fullgraph=False,
        dynamic=None,
    ):
        super().__init__()
        options = {
            "mode": mode,
            "dynamic": dynamic,
            "fullgraph": fullgraph,
        }  # "memory_format": "channels_last"
        self.compile_fn = partial(compile, backend="nexfort", options=options)

    @singledispatchmethod
    def execute(self, model, ckpt_name=None, **kwargs):
        raise NotImplementedError(f"Cannot execute {type(model)=}")

    def _apply_hooks(self, model: ModelPatcher):
        if isinstance(model, ModelPatcher):
            model.model._register_state_dict_hook(state_dict_hook)
            model.model.diffusion_model._register_state_dict_hook(state_dict_hook)

    @execute.register(ModelPatcher)
    @torch.inference_mode()
    def _(self, model, ckpt_name: Optional[str] = None, **kwargs):
        diffusion_model = model.model.diffusion_model
       
        model.model.diffusion_model = apply_memory_format(diffusion_model, torch.channels_last)
        model.model.diffusion_model = self.compile_fn(diffusion_model)
        self._apply_hooks(model)
        model.weight_inplace_update = True
        return model

    @execute.register(VAE)
    @torch.inference_mode()
    def _(self, model, ckpt_name: Optional[str] = None, **kwargs):
        model.first_stage_model.decode = self.compile_fn(model.first_stage_model.decode)
        return model

    @execute.register(ControlNet)
    @torch.inference_mode()
    def _(self, model, ckpt_name: Optional[str] = None, **kwargs):
        torch_model = model.control_model
        torch_model = apply_memory_format(torch_model, torch.channels_last)
        compiled_model: torch.nn.Module = self.compile_fn(torch_model)
        compiled_model._register_state_dict_hook(state_dict_hook)
        model.control_model = compiled_model
        return model

    @execute.register(ControlLora)
    @torch.inference_mode()
    def _(self, model, ckpt_name: Optional[str] = None, **kwargs):
        def compile_cnet(model):
            out: torch.nn.Module = self.compile_fn(model)
            out._register_state_dict_hook(state_dict_hook)
            return out

        model = OneDiffControlLora.from_controllora(model, compile_fn=compile_cnet)
        return model
