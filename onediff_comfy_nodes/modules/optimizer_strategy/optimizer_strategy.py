from abc import ABC, abstractmethod

from comfy import model_management
from comfy.model_patcher import ModelPatcher

from onediff.infer_compiler.with_oneflow_compile import DeployableModule


class OptimizerStrategy(ABC):
    """Interface for optimization strategies."""

    @abstractmethod
    def apply(self, model: ModelPatcher, ckpt_name=""):
        """Apply the optimization strategy to the model."""
        pass


def set_compiled_options(module: DeployableModule, graph_file="unet"):
    assert isinstance(module, DeployableModule)
    compile_options = {
        "graph_file": graph_file,
        "graph_file_device": model_management.get_torch_device(),
    }
    module._deployable_module_options.update(compile_options)
