import torch

from ..deployable_module import DeployableModule


class NexfortDeployableModule(DeployableModule):
    def __init__(self, compiled_module, torch_module):
        torch.nn.Module.__init__(self)
        object.__setattr__(self, "_deployable_module_model", compiled_module)
        object.__setattr__(self, "_modules", compiled_module._modules)
        object.__setattr__(self, "_torch_module", torch_module)

    def __call__(self, *args, **kwargs):
        return self._deployable_module_model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._deployable_module_model, name)
