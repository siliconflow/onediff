import torch
import oneflow as flow
from oneflow.framework.args_tree import ArgsTree
from .log_utils import logger


def input_output_processor(func):
    def process_input(*args, **kwargs):
        def input_fn(value):
            if isinstance(value, torch.Tensor):
                # TODO: https://github.com/siliconflow/sd-team/issues/109
                return flow.utils.tensor.from_torch(value.contiguous())
            else:
                return value

        args_tree = ArgsTree((args, kwargs), False, tensor_type=torch.Tensor)
        input_count = len(
            [v for v in args_tree.iter_nodes() if isinstance(v, torch.Tensor)]
        )
        out = args_tree.map_leaf(input_fn)
        mapped_args = out[0]
        mapped_kwargs = out[1]
        return mapped_args, mapped_kwargs, input_count

    def process_output(output):
        def output_fn(value):
            if isinstance(value, flow.Tensor):
                return flow.utils.tensor.to_torch(value)
            else:
                return value

        out_tree = ArgsTree((output, None), False)
        out = out_tree.map_leaf(output_fn)
        return out[0]

    def wrapper(cls, *args, **kwargs):
        mapped_args, mapped_kwargs, input_count = process_input(*args, **kwargs)
        if cls._deployable_module_use_graph:
            count = getattr(cls, "_deployable_module_input_count", input_count)
            if count != input_count:
                logger.warning(
                    f"{type(cls._deployable_module_model.oneflow_module)} input count changed, will reload graph. "
                    f"When setting the strength parameter in the 'Apply ControlNet' node, please exercise caution. "
                    f"Please check and avoid toggling the strength parameter between values greater than 0 and 0, as it may lead to unnecessary graph reloading."
                )

                cls._deployable_module_dpl_graph = None
                setattr(cls, "_load_graph_first_run", True)

            setattr(cls, "_deployable_module_input_count", input_count)
        output = func(cls, *mapped_args, **mapped_kwargs)
        return process_output(output)

    return wrapper
