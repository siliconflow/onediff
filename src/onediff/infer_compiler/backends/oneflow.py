import torch
from .registry import register_backend


@register_backend("oneflow")
def compile(torch_module: torch.nn.Module, *, use_graph=True, dynamic=True, options={}):
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
    """
    from ..transform.custom_transform import set_default_registry
    from ..oneflow.deployable_module import OneflowDeployableModule
    from ..oneflow.utils import get_mixed_deployable_module

    set_default_registry()

    def wrap_module(module):
        if isinstance(module, OneflowDeployableModule):
            assert not module._is_raw_deployable_module
            return module.__class__.from_existing(module, use_graph, dynamic, options)
        else:
            return get_mixed_deployable_module(module.__class__)(
                module, None, use_graph, dynamic, options
            )

    model = wrap_module(torch_module)
    assert isinstance(model, OneflowDeployableModule)
    assert isinstance(model, torch_module.__class__)

    def state_dict_hook(module, state_dict, prefix, local_metadata):
        pytorch_key_prefix = "_deployable_module_model._torch_module."
        new_state_dict = type(state_dict)()
        for k, v in state_dict.items():
            # _deployable_module_model._torch_module.out.2.weight => out.2.weight
            if k.startswith(pytorch_key_prefix):
                new_k = k[len(pytorch_key_prefix) :]
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    model._register_state_dict_hook(state_dict_hook)

    return model
