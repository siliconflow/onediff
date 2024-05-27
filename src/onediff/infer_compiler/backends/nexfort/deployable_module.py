import torch

from ..deployable_module import DeployableModule


class NexfortDeployableModule(DeployableModule):
    def __init__(self, compiled_module, torch_module):
        torch.nn.Module.__init__(self)
        object.__setattr__(self, "_deployable_module_model", compiled_module)
        object.__setattr__(self, "_modules", compiled_module._orig_mod._modules)
        object.__setattr__(self, "_parameters", compiled_module._orig_mod._parameters)
        object.__setattr__(self, "_buffers", compiled_module._orig_mod._buffers)
        # object.__setattr__(self, "_torch_module", torch_module) _orig_mod

    def __call__(self, *args, **kwargs):
        return self._deployable_module_model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._deployable_module_model, name)


def get_mixed_deployable_module(module_cls) -> type:
    class MixedNexfortDeployableModule(NexfortDeployableModule, module_cls):
        def __init__(self, compiled_module, torch_module):
            super().__init__(compiled_module, torch_module)

        def _get_name(self):
            return f"{self.__class__.__name__}(of {module_cls.__name__})"

    return MixedNexfortDeployableModule
