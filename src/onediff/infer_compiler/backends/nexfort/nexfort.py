import os
import logging
import dataclasses
from pathlib import Path

import torch
from ..registry import register_backend

logging.basicConfig(level=logging.INFO)

@register_backend("nexfort")
def compile(torch_module: torch.nn.Module, *, options=None):
    from nexfort.utils.memory_format import apply_memory_format
    from nexfort.compilers import nexfort_compile
    from .deployable_module import NexfortDeployableModule
    from ..options import CompileOptions

    options = options if options is not None else CompileOptions()
    nexfort_options = options.nexfort
    compiled_model = nexfort_compile(torch_module, **nexfort_options)

    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if cache_dir and not Path(cache_dir).exists():
        logging.info(f"Enabled Inductor - Autotuning Cache for {torch_module.__class__.__name__}")

    # return NexfortDeployableModule(compiled_model, torch_module)
    return compiled_model
