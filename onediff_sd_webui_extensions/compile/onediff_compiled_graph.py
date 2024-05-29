import dataclasses
import torch
from onediff.infer_compiler import DeployableModule
from modules import sd_models_types


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
        graph_module: DeployableModule = None,
        quantized=False,
    ):
        if sd_model is None:
            return
        self.name = sd_model.sd_checkpoint_info.name
        self.filename = sd_model.sd_checkpoint_info.filename
        self.sha = sd_model.sd_model_hash
        self.eager_module = sd_model.model.diffusion_model
        self.graph_module = graph_module
        self.quantized = quantized
