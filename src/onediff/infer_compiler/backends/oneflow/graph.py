import oneflow as flow

from onediff.utils import logger
from .transform.manager import transform_mgr
from .transform.builtin_transform import reverse_proxy_class
from .utils.cost_util import cost_cnt


class OneflowGraph(flow.nn.Graph):
    @flow.nn.Graph.with_dynamic_input_shape()
    def __init__(self, model):
        super().__init__(enable_get_runtime_state_dict=True)
        self.model = model
        logger.info(f"Building a graph for {model.__class__.__name__} ...")
        # self.config.enable_cudnn_conv_heuristic_search_algo(False)
        self.config.allow_fuse_add_to_output(True)

    def build(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @cost_cnt(transform_mgr.debug_mode)
    def load_graph(self, file_path, device=None, run_warmup=True, *, state_dict=None):
        state_dict = state_dict if state_dict is not None else flow.load(file_path)
        self.graph_state_dict = state_dict  # used for OneflowGraph.save_graph

        if device is not None:
            state_dict = flow.nn.Graph.runtime_state_dict_to(state_dict, device)

        self.load_runtime_state_dict(state_dict, warmup_with_run=run_warmup)

    @cost_cnt(transform_mgr.debug_mode)
    def save_graph(self, file_path, *, process_state_dict: lambda x: x):
        try:

            if hasattr(self, "graph_state_dict"):
                logger.debug(f"debug ci,")
                flow.save(self.graph_state_dict, file_path)
                return
            logger.debug(f"debug ci,")
            state_dict = self.runtime_state_dict()
            logger.debug(f"debug ci,")
            import oneflow.framework.args_tree as args_tree

            logger.debug(f"debug ci,")

            def disabled_dataclass(value):
                return False

            logger.debug(f"debug ci,")
            original_is_dataclass = args_tree._is_dataclass
            args_tree._is_dataclass = disabled_dataclass

            import dataclasses

            def reverse_dataclass(value):
                if dataclasses.is_dataclass(value):
                    return reverse_proxy_class(type(value))(**value)
                else:
                    return value

            for name, rsd in state_dict.items():
                output = state_dict[name]["outputs_original"]
                out_tree = args_tree.ArgsTree((output, None), False)
                # dataclass type needs to be reversed to torch type to avoid saving error.
                out = out_tree.map_leaf(reverse_dataclass)
                state_dict[name]["outputs_original"] = out[0]
            logger.debug(f"debug ci,")
            args_tree._is_dataclass = original_is_dataclass
            logger.debug(f"debug ci,")
            state_dict = process_state_dict(state_dict)
            logger.debug(f"debug ci,")
            flow.save(state_dict, file_path)
        except Exception as e:
            import traceback

            print(f"{e=}")
            print(traceback.format_exc())
