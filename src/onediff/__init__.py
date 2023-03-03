__version__ = "0.5.0"
__author__ = "OneFlow"
__credits__ = "OneFlow contributors"
import oneflow as flow


# monkey patch hacks
flow.strided = None
flow_randn = flow.randn


def dummy_randn(*args, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if not k == "layout"}
    return flow_randn(*args, **kwargs)


flow.randn = dummy_randn


flow.mock_torch.enable()
from .pipeline_stable_diffusion_oneflow import OneFlowStableDiffusionPipeline
from .pipeline_alt_diffusion_oneflow import OneFlowAltDiffusionPipeline
from .pipeline_stable_diffusion_inpaint_oneflow import (
    OneFlowStableDiffusionInpaintPipeline,
)
