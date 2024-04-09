import dataclasses
from typing import Dict
import torch


@dataclasses.dataclass
class OneflowCompileOptions:
    use_graph: bool = True
    debug_level: int = -1
    max_cached_graph_size: int = 9
    graph_file: str = None
    graph_file_device: torch.device = None


@dataclasses.dataclass
class NexfortInductorCompileOptions:
    disable: bool = False
    mode: str = None
    options: Dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class NexfortCompileOptions:
    memory_format: torch.memory_format
    fuse_qkv_projections: bool
    inductor: NexfortInductorCompileOptions

    def __init__(
        self,
        memory_format=torch.channels_last,
        fuse_qkv_projections=True,
        inductor=None,
    ):
        if isinstance(memory_format, str):
            memory_format = getattr(torch, memory_format)
        self.memory_format = memory_format
        self.fuse_qkv_projections = fuse_qkv_projections
        self.inductor = (
            inductor if inductor is not None else NexfortInductorCompileOptions()
        )


@dataclasses.dataclass
class CompileOptions:
    # common options
    dynamic: bool

    # oneflow specific options
    oneflow: OneflowCompileOptions

    # nexfort specific options
    nexfort: NexfortCompileOptions

    def __init__(self, dynamic=True, oneflow=None, nexfort=None):
        self.dynamic = dynamic
        self.oneflow = oneflow if oneflow is not None else OneflowCompileOptions()
        self.nexfort = nexfort if nexfort is not None else NexfortCompileOptions()
