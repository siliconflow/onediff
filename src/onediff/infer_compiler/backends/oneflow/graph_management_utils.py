import importlib
import os
from typing import Dict

import torch
import oneflow as flow  # usort: skip
from functools import wraps
from pathlib import Path

from oneflow.framework.args_tree import ArgsTree

from onediff.utils import logger
from ..env_var import OneflowCompileOptions
from .transform.builtin_transform import torch2oflow
from .transform.manager import transform_mgr
from .utils.cost_util import cost_time
from .utils.hash_utils import generate_input_structure_key, generate_model_structure_key


def _prepare_file_path(file_path):
    if isinstance(file_path, Path):
        file_path = str(file_path)
    if file_path.endswith(".graph"):
        file_path = file_path[:-6]
    return file_path


@cost_time(debug=transform_mgr.debug_mode, message="generate graph file name")
def generate_graph_file_name(file_path, deployable_module, args, kwargs):
    file_path = _prepare_file_path(file_path)
    args_tree = ArgsTree((args, kwargs), gen_name=False, tensor_type=torch.Tensor)
    input_structure_key = generate_input_structure_key(args_tree)
    model_structure_key = generate_model_structure_key(deployable_module)
    # Combine cache keys
    cache_key = f"{input_structure_key}_{model_structure_key}"
    return f"{file_path}_{cache_key}.graph"


def graph_file_management(func):
    @wraps(func)
    def wrapper(self: "OneflowDeployableModule", *args, **kwargs):
        compile_options = (
            self._deployable_module_options
            if hasattr(self, "_deployable_module_options")
            else OneflowCompileOptions()
        )
        graph_file = compile_options.graph_file
        is_first_load = self._load_graph_first_run and graph_file is not None

        if self._deployable_module_input_structure_key is None:
            args_tree = ArgsTree(
                (args, kwargs), gen_name=False, tensor_type=torch.Tensor
            )
            self._deployable_module_input_structure_key = generate_input_structure_key(
                args_tree
            )

        if is_first_load:
            self._load_graph_first_run = False
            input_structure_key = self._deployable_module_input_structure_key
            model_structure_key = generate_model_structure_key(deployable_module=self)
            file_path = _prepare_file_path(graph_file)
            graph_file = (
                f"{file_path}_{input_structure_key}_{model_structure_key}.graph"
            )

        def process_state_dict_before_saving(state_dict: Dict):
            nonlocal self, args, kwargs, graph_file
            graph = self.get_graph()
            if importlib.util.find_spec("register_comfy"):
                from register_comfy import CrossAttntionStateDictPatch as state_patch

                state_dict = state_patch.process_state_dict_before_saving(
                    state_dict, graph=graph
                )
            return state_dict

        def handle_graph_loading():
            nonlocal graph_file, compile_options, is_first_load
            if not is_first_load:
                return

            if not os.path.exists(graph_file):
                logger.info(
                    f"Graph file {graph_file} does not exist! Generating graph."
                )
            else:
                graph_device = compile_options.graph_file_device
                state_dict = flow.load(graph_file)
                self.load_graph(
                    graph_file, torch2oflow(graph_device), state_dict=state_dict
                )
                logger.info(f"Loaded graph file: {graph_file}")
                is_first_load = False

        def handle_graph_saving():
            nonlocal graph_file, compile_options, is_first_load
            if not is_first_load:
                return

            parent_dir = os.path.dirname(graph_file)
            if parent_dir != "":
                os.makedirs(parent_dir, exist_ok=True)

            # Avoid graph file conflicts
            if os.path.exists(graph_file):
                raise FileExistsError(f"File {graph_file} exists!")
            try:

                self.save_graph(
                    graph_file, process_state_dict=process_state_dict_before_saving
                )
                logger.info(f"Saved graph file: {graph_file}")

            except Exception as e:
                logger.error(f"Failed to save graph file: {graph_file}! {e}")

        if self._deployable_module_options.use_graph and is_first_load:
            handle_graph_loading()
            ret = func(self, *args, **kwargs)
            handle_graph_saving()
        else:
            ret = func(self, *args, **kwargs)

        return ret

    return wrapper
