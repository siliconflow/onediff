import torch
from .deployable_module import DeployableModule


def onediff_compile(
    torch_module: torch.nn.Module,
    *,
    backend="oneflow",
    use_graph=True,
    dynamic=True,
    options={},
) -> DeployableModule:
    from .backends.registry import lookup_backend

    backend = lookup_backend(backend)
    model = backend(torch_module, use_graph=True, dynamic=True, options=options)
    return model
