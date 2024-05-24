import dataclasses
import torch
from ..registry import register_backend


@register_backend("nexfort")
def compile(torch_module: torch.nn.Module, *, options=None):
    from nexfort.utils.memory_format import apply_memory_format
    from nexfort.compilers import nexfort_compile
    from .deployable_module import NexfortDeployableModule

    nexfort_options = options if options is not None else dict()
    compiled_model = nexfort_compile(torch_module, **nexfort_options)
    # return NexfortDeployableModule(compiled_model, torch_module)
    return compiled_model
