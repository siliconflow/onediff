import torch
from .deployable_module import DeployableModule

_DEFAULT_BACKEND = "oneflow"

def compile(
    torch_module: torch.nn.Module, *, backend=_DEFAULT_BACKEND, options=None
) -> DeployableModule:
    from ..backends.registry import lookup_backend

    backend = lookup_backend(backend)
    model = backend(torch_module, options=options)
    return model


def oneflow_compile(torch_module: torch.nn.Module, *, options=None) -> DeployableModule:
    return compile(torch_module, backend="oneflow", options=options)
