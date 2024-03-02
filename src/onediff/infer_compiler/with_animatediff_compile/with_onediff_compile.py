import torch 
from .deployable_module import DeployableModule
from .oneflow_compiler import DualModule
from ..transform.custom_transform import set_default_registry


def oneflow_compile(
    torch_module: torch.nn.Module, *, use_graph=True, dynamic=True, options={},
) -> DeployableModule:
    """
    Transform a torch nn.Module to oneflow.nn.Module, then optimize it with oneflow.nn.Graph.
    Args:
       model (torch.nn.Module): Module to optimize
       use_graph (bool): Whether to optimize with oneflow.nn.Graph
       dynamic (bool): When this is True, we will generate one graph and reuse it to avoid recompilations when
        input shape change.  This may not always work as some operations/optimizations break the contition of
        reusing.  When this is False, we will generate a graph for each new input shape, and will always specialize.
        By default (True).
       options (dict): A dictionary of options to pass to the compiler:
        - 'debug' which config the nn.Graph debug level, default -1(no debug info), max 3(max debug info);
        - 'size' which config the cache size when cache is enabled. Note that after onediff v0.12, cache is default disabled.
        - 'graph_file' (None) generates a compilation cache file. If the file exists, loading occurs; if not, the compilation result is saved after the first run.
        - 'graph_file_device' (None) sets the device for the graph file, default None.  If set, the compilation result will be converted to the specified device.
        - 'sync_with_oneflow_only' (True) only sync with oneflow, default True. If set False, the parameters will be synced with oneflow and torch module.
    """

    set_default_registry()
    if isinstance(torch_module, DeployableModule):
        return torch_module
    compiled_model = DualModule(torch_module, use_graph, dynamic, options)
    out =  DeployableModule(torch_module, compiled_model)
    assert isinstance(out, torch_module.__class__)
    assert out.state_dict().keys() == torch_module.state_dict().keys()
    return out
