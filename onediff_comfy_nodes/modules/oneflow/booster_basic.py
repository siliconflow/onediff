from functools import singledispatchmethod
from typing import Optional

import torch
from comfy import model_management
from comfy.controlnet import ControlLora, ControlNet
from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE
from onediff.infer_compiler import oneflow_compile
from onediff.infer_compiler.backends.oneflow import (
    OneflowDeployableModule as DeployableModule,
)

from ..booster_interface import BoosterExecutor
from .onediff_controlnet import OneDiffControlLora
from .utils.booster_utils import (
    get_model_type,
    is_fp16_model,
    set_compiled_options,
    set_environment_for_svd_img2vid,
)
from .utils.graph_path import generate_graph_path


class BasicOneFlowBoosterExecutor(BoosterExecutor):
    @singledispatchmethod
    def execute(self, model, ckpt_name=None, **kwargs):
        raise NotImplementedError(f"Cannot execute {type(model)=}")

    @execute.register
    def _(self, model: ModelPatcher, ckpt_name: Optional[str] = None, **kwargs):
        torch_model = model.model.diffusion_model
        if isinstance(torch_model, DeployableModule):
            return model

        set_environment_for_svd_img2vid(model)

        if not is_fp16_model(torch_model):
            print(
                f"Warning: Model {type(torch_model)} is not an FP16 model. Compilation will be skipped!"
            )
            return model

        compiled_model = oneflow_compile(torch_model)

        model.model.diffusion_model = compiled_model

        graph_file = generate_graph_path(f"{type(model).__name__}", model=model.model)
        set_compiled_options(compiled_model, graph_file)

        model.weight_inplace_update = True
        return model

    @execute.register(ControlNet)
    def _(self, model, ckpt_name: Optional[str] = None, **kwargs) -> ControlNet:
        torch_model = model.control_model
        if isinstance(torch_model, DeployableModule):
            return model

        if not is_fp16_model(torch_model):
            type_set = get_model_type(torch_model)
            print(
                f"Warning: Model {type(torch_model)} with parameter types {type_set} is not an FP16 model. Compilation will be skipped!"
            )
            return model

        compiled_model = oneflow_compile(torch_model)
        model.control_model = compiled_model

        graph_file = generate_graph_path(ckpt_name, torch_model)
        set_compiled_options(compiled_model, graph_file)
        return model

    @execute.register(VAE)
    def _(self, model, ckpt_name: Optional[str] = None, **kwargs) -> VAE:
        torch_model = model.first_stage_model
        if isinstance(torch_model, DeployableModule):
            return model

        device = model_management.get_torch_device()
        gpu_name = torch.cuda.get_device_name(device)

        if gpu_name == "NVIDIA A800-SXM4-80GB":
            # TODO Record the problem to issues
            model_type = type(torch_model)
            print(
                f"Warning: Model {model_type} not supported on {gpu_name}. Compilation skipped!"
            )
            return model

        compiled_model = oneflow_compile(torch_model)
        model.first_stage_model = compiled_model

        graph_file = generate_graph_path(ckpt_name, torch_model)
        set_compiled_options(compiled_model, graph_file)
        return model

    @execute.register(ControlLora)
    def _(self, model, ckpt_name: Optional[str] = None, **kwargs):
        def gen_compile_options(model):

            graph_file = generate_graph_path(ckpt_name, model)
            return {
                "graph_file": graph_file,
                "graph_file_device": model_management.get_torch_device(),
            }

        controlnet = OneDiffControlLora.from_controllora(
            model, gen_compile_options=gen_compile_options
        )

        return controlnet
