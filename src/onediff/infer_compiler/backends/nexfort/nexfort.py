import os
import dataclasses
from pathlib import Path

import torch
from ..registry import register_backend


@register_backend("nexfort")
def compile(torch_module: torch.nn.Module, *, options=None):
    from nexfort.compilers import nexfort_compile
    from nexfort.utils.logging import logger
    from .deployable_module import NexfortDeployableModule
    from ..options import CompileOptions

        # TODO(): using jsonschema to define the options schema
        options = json.loads(options)

    nexfort_options = options if options is not None else dict()
    compiled_model = nexfort_compile(torch_module, **nexfort_options)

    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if cache_dir and not Path(cache_dir).exists():
        logger.info(f"Enabled Inductor - Autotuning Cache for {torch_module.__class__.__name__}")

    # return NexfortDeployableModule(compiled_model, torch_module)
    return compiled_model
