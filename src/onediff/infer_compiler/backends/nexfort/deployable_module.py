from types import FunctionType
from typing import Type, Union

import torch
from torch import nn

from ..deployable_module import DeployableModule

DISABLE_DEPLOYABLE = False


def fallback_to_eager(value: bool = False):
    """
    Set the DISABLE_DEPLOYABLE environment variable.

    Args:
        value (bool): The value to set for the environment variable.
    """
    global DISABLE_DEPLOYABLE
    DISABLE_DEPLOYABLE = not value


class NexfortDeployableModule(DeployableModule):
    def __init__(self, compiled_module, torch_module):
        torch.nn.Module.__init__(self)
        object.__setattr__(self, "_torch_module", torch_module)
        object.__setattr__(self, "_deployable_module_model", compiled_module)
        # https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/eval_frame.py#L148
        if isinstance(torch_module, nn.Module) and isinstance(
            compiled_module, torch._dynamo.eval_frame.OptimizedModule
        ):
            object.__setattr__(self, "_modules", compiled_module._orig_mod._modules)
            object.__setattr__(
                self, "_parameters", compiled_module._orig_mod._parameters
            )
            object.__setattr__(self, "_buffers", compiled_module._orig_mod._buffers)

    def forward(self, *args, **kwargs):
        if DISABLE_DEPLOYABLE:
            return self._torch_module(*args, **kwargs)
        with torch._dynamo.utils.disable_cache_limit():
            return self._deployable_module_model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._deployable_module_model, name)


def _create_deployable_function(
    compiled_model, torch_module: FunctionType = None
) -> FunctionType:
    def deploy_function(*args, **kwargs):
        if DISABLE_DEPLOYABLE:
            return torch_module(*args, **kwargs)
        with torch._dynamo.utils.disable_cache_limit():
            return compiled_model(*args, **kwargs)

    return deploy_function


def _create_mixed_deployable_module(
    compiled_model, torch_module: nn.Module
) -> Type[NexfortDeployableModule]:
    module_cls = type(torch_module)

    class MixedNexfortDeployableModule(NexfortDeployableModule, module_cls):
        def __init__(self, compiled_module, torch_module):
            super().__init__(compiled_module, torch_module)

        def _get_name(self):
            return f"{self.__class__.__name__}(of {module_cls.__name__})"

    return MixedNexfortDeployableModule(
        compiled_module=compiled_model, torch_module=torch_module
    )


def get_deployable_module(
    torch_module: Union[nn.Module, FunctionType], compiled_model
) -> Union[Type[NexfortDeployableModule], FunctionType]:
    if not isinstance(torch_module, nn.Module):
        return _create_deployable_function(compiled_model, torch_module)
    return _create_mixed_deployable_module(compiled_model, torch_module)
