import dataclasses
from typing import Union

import torch
from ldm.modules.diffusionmodules.openaimodel import UNetModel as LdmUNetModel
from modules import sd_models_types, shared

from onediff.infer_compiler import DeployableModule
from sgm.modules.diffusionmodules.openaimodel import UNetModel as SgmUNetModel

from .backend import OneDiffBackend


def disable_unet_checkpointing(
    unet_model: Union[LdmUNetModel, SgmUNetModel]
) -> Union[LdmUNetModel, SgmUNetModel]:
    from ldm.modules.attention import BasicTransformerBlock as LdmBasicTransformerBlock
    from ldm.modules.diffusionmodules.openaimodel import ResBlock as LdmResBlock
    from sgm.modules.attention import BasicTransformerBlock as SgmBasicTransformerBlock
    from sgm.modules.diffusionmodules.openaimodel import ResBlock as SgmResBlock

    for module in unet_model.modules():
        if isinstance(module, (LdmBasicTransformerBlock, SgmBasicTransformerBlock)):
            module.checkpoint = False
        if isinstance(module, (LdmResBlock, SgmResBlock)):
            module.use_checkpoint = False
    return unet_model


def get_onediff_backend() -> OneDiffBackend:
    return OneDiffBackend(shared.opts.onediff_compiler_backend)


def is_oneflow_backend(backend: Union[OneDiffBackend, None] = None) -> bool:
    return (backend or get_onediff_backend()) == OneDiffBackend.ONEFLOW


def is_nexfort_backend(backend: Union[OneDiffBackend, None] = None) -> bool:
    return (backend or get_onediff_backend()) == OneDiffBackend.NEXFORT


def init_backend(backend: Union[OneDiffBackend, None] = None):
    backend = backend or get_onediff_backend()
    if is_oneflow_backend(backend):
        from .oneflow.utils import init_oneflow_backend

        init_oneflow_backend()
    elif is_nexfort_backend(backend):
        from .nexfort.utils import init_nexfort_backend

        init_nexfort_backend()
    else:
        raise NotImplementedError(f"invalid backend {backend}")


@dataclasses.dataclass
class OneDiffCompiledGraph:
    name: str = None
    filename: str = None
    sha: str = None
    eager_module: torch.nn.Module = None
    graph_module: DeployableModule = None
    quantized: bool = False

    def __init__(
        self,
        sd_model: sd_models_types.WebuiSdModel = None,
        unet_model=None,
        graph_module: DeployableModule = None,
        quantized=False,
    ):
        if sd_model is None:
            return
        self.name = sd_model.sd_checkpoint_info.name
        self.filename = sd_model.sd_checkpoint_info.filename
        self.sha = sd_model.sd_model_hash
        self.eager_module = unet_model or sd_model.model.diffusion_model
        self.graph_module = graph_module
        self.quantized = quantized
