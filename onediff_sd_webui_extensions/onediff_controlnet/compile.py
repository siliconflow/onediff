from functools import wraps

import onediff_shared
from compile.utils import (
    OneDiffCompiledGraph,
    disable_unet_checkpointing,
    get_onediff_backend,
    is_nexfort_backend,
    is_oneflow_backend,
)
from modules import shared

from onediff.infer_compiler import oneflow_compile

from .hijack import hijack_controlnet_extension
from .utils import check_if_controlnet_enabled


def onediff_controlnet_decorator(func):
    @wraps(func)
    # TODO: restore hijacked func here
    def wrapper(self, p, *arg, **kwargs):
        try:
            onediff_shared.controlnet_enabled = check_if_controlnet_enabled(p)
            if onediff_shared.controlnet_enabled:
                hijack_controlnet_extension(p)
            return func(self, p, *arg, **kwargs)
        finally:
            if onediff_shared.controlnet_enabled:
                onediff_shared.previous_is_controlnet = True
            else:
                onediff_shared.controlnet_compiled = False
                onediff_shared.previous_is_controlnet = False

    return wrapper


def compile_controlnet_ldm_unet(sd_model, unet_model, *, backend=None, options=None):
    backend = backend or get_onediff_backend()

    if is_oneflow_backend():
        disable_unet_checkpointing(unet_model)
        compiled_model = oneflow_compile(unet_model, options=options)
    elif is_nexfort_backend():
        raise NotImplementedError(
            "nexfort backend for controlnet is not implemented yet"
        )
    # TODO: refine here
    compiled_graph = OneDiffCompiledGraph(sd_model, compiled_model)
    compiled_graph.eager_module = unet_model
    compiled_graph.name += "_controlnet"
    return compiled_graph
