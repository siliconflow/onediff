import hashlib
import importlib
import os
from typing import Dict
import torch
import oneflow as flow
from pathlib import Path
from functools import wraps
from oneflow.framework.args_tree import ArgsTree
from ..transform.builtin_transform import torch2oflow
from ..transform.manager import transform_mgr
from .log_utils import logger
from .cost_util import cost_time
from .options import OneflowCompileOptions


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
        compile_options = (
            self._deployable_module_options
            if hasattr(self, "_deployable_module_options")
            else OneflowCompileOptions()
        )
        graph_file = compile_options.graph_file

        is_first_load = (
            getattr(self, "_load_graph_first_run", True) and graph_file is not None
        )

        if is_first_load:
            graph_file = generate_graph_file_name(
                graph_file, self, args=args, kwargs=kwargs
            )
            setattr(self, "_load_graph_first_run", False)
            # Avoid graph file conflicts
            if importlib.util.find_spec("register_comfy"):
                from register_comfy import CrossAttntionStateDictPatch as state_patch
                attn2_patch_sum = state_patch.attn2_patch_sum(input_kwargs=kwargs)
                if attn2_patch_sum > 0:
                    graph_file = graph_file.replace(".graph", f"_attn2_{attn2_patch_sum}.graph")

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
            try:
                parent_dir = os.path.dirname(graph_file)
                if parent_dir != "":
                    os.makedirs(parent_dir, exist_ok=True)
                
                # Avoid graph file conflicts
                if os.path.exists(graph_file):
                    raise FileExistsError(f"File {graph_file} exists!")

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
