import torch
from .deployable_module import DeployableModule


def compile(
    torch_module: torch.nn.Module,
    *,
    backend="nexfort",
    use_graph=True,
    dynamic=True,
    options={},
) -> DeployableModule:
    from .backends.registry import lookup_backend

    backend = lookup_backend(backend)
    model = backend(torch_module, use_graph=True, dynamic=True, options=options)
    return model


def oneflow_compile(
    torch_module: torch.nn.Module, *, use_graph=True, dynamic=True, options={},
) -> DeployableModule:
    return compile(
        torch_module,
        backend="oneflow",
        use_graph=use_graph,
        dynamic=dynamic,
        options=options,
    )
