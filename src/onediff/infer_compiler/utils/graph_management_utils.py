import os
import torch
import oneflow as flow
from pathlib import Path
from functools import wraps
from oneflow.framework.args_tree import ArgsTree
from ..transform.builtin_transform import torch2oflow
from .log_utils import logger


def graph_file_management(func):
    @wraps(func)
    def wrapper(self: "DeployableModule", *args, **kwargs):
        compile_options = getattr(self, "_deployable_module_options", {})
        graph_file = compile_options.get("graph_file")

        def generate_graph_file_name(file_path):
            if isinstance(file_path, Path):
                file_path = str(file_path)

            if file_path.endswith(".graph"):
                file_path = file_path[:-6]

            args_tree = ArgsTree((args, kwargs), False, tensor_type=torch.Tensor)
            count = len(
                [v for v in args_tree.iter_nodes() if isinstance(v, flow.Tensor)]
            )
            return f"{graph_file}_{count}.graph"

        if graph_file:
            graph_file = generate_graph_file_name(graph_file)

        def handle_graph_loading():
            nonlocal graph_file, compile_options
            if graph_file is None:
                return

            if not os.path.exists(graph_file):
                logger.info(
                    f"Graph file {graph_file} does not exist! Generating graph."
                )
            else:
                graph_device = compile_options.get("graph_file_device")
                self.load_graph(graph_file, torch2oflow(graph_device))
                logger.info(f"Loaded graph file: {graph_file}")
                compile_options["graph_file"] = None
                graph_file = None

        def handle_graph_saving():
            nonlocal graph_file, compile_options
            if graph_file is None:
                return

            try:
                parent_dir = os.path.dirname(graph_file)
                if parent_dir != "":
                    os.makedirs(parent_dir, exist_ok=True)

                self.save_graph(graph_file)
                logger.info(f"Saved graph file: {graph_file}")

            except Exception as e:
                logger.error(f"Failed to save graph file: {graph_file}! {e}")

            finally:
                compile_options["graph_file"] = None

        handle_graph_loading()
        ret = func(self, *args, **kwargs)
        handle_graph_saving()

        return ret

    return wrapper
