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
    inductor: NexfortInductorCompileOptions

    def __init__(self):
        self.inductor = NexfortInductorCompileOptions()


@dataclasses.dataclass
class CompileOptions:
    # common options
    dynamic: bool

    # oneflow specific options
    oneflow: OneflowCompileOptions

    # nexfort specific options
    nexfort: NexfortCompileOptions

    def __init__(self, dynamic=True):
        self.dynamic = dynamic
        self.oneflow = OneflowCompileOptions()
        self.nexfort = NexfortCompileOptions()
