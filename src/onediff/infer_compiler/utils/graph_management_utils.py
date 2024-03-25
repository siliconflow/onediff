import hashlib
import os
import torch
import oneflow as flow
from pathlib import Path
from functools import wraps
from oneflow.framework.args_tree import ArgsTree
from ..transform.builtin_transform import torch2oflow
from ..transform.manager import transform_mgr
from .log_utils import logger
from .cost_util import cost_time


def calculate_model_hash(model):
    return hashlib.sha256(f"{model}".encode("utf-8")).hexdigest()


@cost_time(debug=transform_mgr.debug_mode, message="generate graph file name")
def generate_graph_file_name(file_path, deployable_module, args, kwargs):
    if isinstance(file_path, Path):
        file_path = str(file_path)

    if file_path.endswith(".graph"):
        file_path = file_path[:-6]

    args_tree = ArgsTree((args, kwargs), False, tensor_type=torch.Tensor)
    count = len([v for v in args_tree.iter_nodes() if isinstance(v, flow.Tensor)])

    model = deployable_module._deployable_module_model.oneflow_module

    cache_key = calculate_model_hash(model) + "_" + flow.__version__
    return f"{file_path}_{count}_{cache_key}.graph"


def graph_file_management(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        compile_options = getattr(self, "_deployable_module_options", {})
        graph_file = compile_options.get("graph_file", None)

        is_first_load = (
            getattr(self, "_load_graph_first_run", True) and graph_file is not None
        )

        if is_first_load:
            graph_file = generate_graph_file_name(
                graph_file, self, args=args, kwargs=kwargs
            )
            setattr(self, "_load_graph_first_run", False)

        def handle_graph_loading():
            nonlocal graph_file, compile_options, is_first_load
            if not is_first_load:
                return

            if not os.path.exists(graph_file):
                logger.info(
                    f"Graph file {graph_file} does not exist! Generating graph."
                )
            else:
                graph_device = compile_options.get("graph_file_device")
                self.load_graph(graph_file, torch2oflow(graph_device))
                logger.info(f"Loaded graph file: {graph_file}")
                is_first_load = False

        def handle_graph_saving():
            nonlocal graph_file, compile_options, is_first_load
            if not is_first_load:
                return
            try:
                parent_dir = os.path.dirname(graph_file)
                if parent_dir != "":
                    os.makedirs(parent_dir, exist_ok=True)

                self.save_graph(graph_file)
                logger.info(f"Saved graph file: {graph_file}")

            except Exception as e:
                logger.error(f"Failed to save graph file: {graph_file}! {e}")

        if self._deployable_module_use_graph and is_first_load:
            handle_graph_loading()
            ret = func(self, *args, **kwargs)
            handle_graph_saving()
        else:
            ret = func(self, *args, **kwargs)

        return ret

    return wrapper
