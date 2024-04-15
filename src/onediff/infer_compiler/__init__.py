import os
import torch

from .core import *
from .utils import set_default_env_vars
from .utils.options import CompileOptions
from .utils.options import _GLOBAL_compile_options as compile_options


set_default_env_vars()


def oneflow_backend(gm, example_inputs, *args, **kwargs):
    import oneflow as flow
    from oneflow.framework.args_tree import ArgsTree

    with_interp = os.getenv(
        "ONEDIFF_INFER_COMPILER_USE_INTERPRETER", "False"
    ).lower() in ("true", "1", "t",)
    if not with_interp:
        transformed_fn = fx_node_tranform(gm)

    def wrapped_forward(*args, **kwargs):
        def input_fn(value):
            if isinstance(value, torch.Tensor):
                return flow.utils.tensor.from_torch(value.contiguous())
            else:
                return value

        args_tree = ArgsTree((args, kwargs), False, tensor_type=torch.Tensor)
        out = args_tree.map_leaf(input_fn)
        args = out[0]
        if with_interp:
            output = OneFlowInterpreter(gm, garbage_collect_values=False).run(
                *args, **kwargs
            )
        else:
            output = transformed_fn(*args, **kwargs)
        if isinstance(output, tuple):
            return tuple(flow.utils.tensor.to_torch(i) for i in output)
        return flow.utils.tensor.to_torch(output)

    return wrapped_forward
