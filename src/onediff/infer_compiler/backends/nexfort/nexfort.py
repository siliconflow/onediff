from typing import Callable

import torch

from ..registry import register_backend
from .deployable_module import get_deployable_module, NexfortDeployableModule


@register_backend("nexfort")
def compile(torch_module: torch.nn.Module, *, options=None):
    from nexfort.compilers import nexfort_compile

    # Decorator mode
    if torch_module is None:

        def fn(torch_module: Callable):
            if torch_module is None:
                raise RuntimeError("torch_module can't be None")
            return compile(torch_module, options=options)

        return fn

    if isinstance(torch_module, NexfortDeployableModule):
        return compile(torch_module._torch_module, options=options)

    if isinstance(options, str):
        import json

        # TODO(): using jsonschema to define the options schema
        options = json.loads(options)

    nexfort_options = options if options is not None else dict()

    compiled_model = nexfort_compile(torch_module, **nexfort_options)

    return get_deployable_module(torch_module, compiled_model)
