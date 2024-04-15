import dataclasses
import torch
from .registry import register_backend


def make_inductor_options(options):
    inductor_options = {}
    if options is None:
        return inductor_options
    for filed in dataclasses.fields(options):
        filed_name = filed.name
        inductor_options[f"inductor.{filed_name}"] = getattr(options, filed_name)
    return inductor_options


@register_backend("nexfort")
def compile(torch_module: torch.nn.Module, *, options=None):
    from nexfort.utils.memory_format import apply_memory_format
    from nexfort.compilers import nexfort_compile
    from ..nexfort.deployable_module import NexfortDeployableModule
    from ..utils import CompileOptions

    options = options if options is not None else CompileOptions()
    nexfort_options = options.nexfort
    if nexfort_options.memory_format != torch.preserve_format:
        model = apply_memory_format(
            torch_module, memory_format=nexfort_options.memory_format
        )
    model = nexfort_compile(
        model, options=make_inductor_options(nexfort_options.inductor)
    )
    return NexfortDeployableModule(model)
