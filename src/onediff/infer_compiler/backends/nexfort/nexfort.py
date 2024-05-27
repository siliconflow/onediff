import dataclasses
import torch
from ..registry import register_backend


@register_backend("nexfort")
def compile(torch_module: torch.nn.Module, *, options=None):
    from nexfort.compilers import nexfort_compile
    if isinstance(options, str):
        import json

        # TODO(): using jsonschema to define the options schema
        options = json.loads(options)

    nexfort_options = options if options is not None else dict()
    compiled_model = nexfort_compile(torch_module, **nexfort_options)
    return compiled_model
