from functools import wraps

import onediff_shared
from compile import get_compiled_graph
from compile.utils import is_nexfort_backend, is_oneflow_backend

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
    if is_oneflow_backend():
        from compile.oneflow.mock.controlnet import OneFlowOnediffControlNetModel

        from onediff.infer_compiler.backends.oneflow.transform import register

        from .model import OnediffControlNetModel

        register(
            package_names=["scripts.hook"],
            torch2oflow_class_map={
                OnediffControlNetModel: OneFlowOnediffControlNetModel,
            },
        )
    elif is_nexfort_backend():
        # nothing need to do
        pass
    compiled_graph = get_compiled_graph(
        sd_model, unet_model, backend=backend, options=options
    )
    compiled_graph.name += "_controlnet"
    return compiled_graph
