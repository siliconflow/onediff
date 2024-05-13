import hashlib
import os
from typing import Dict
import torch
import oneflow as flow
import pickle
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

        def apply_patch_after_loading_state_dict():
            nonlocal self, args, kwargs
            graph = self.get_graph()

            try:
                attn2_patches = (
                    kwargs.get("transformer_options", {})
                    .get("patches_replace", {})
                    .get("attn2")
                )
                if attn2_patches:
                    patch_file_name = f"{Path(graph_file).name}.patch"
                    new_file_path = Path(graph_file).with_name(patch_file_name)

                    with open(new_file_path, "rb") as fp:
                        idx_mapping = pickle.load(fp)

                    for k, attn_module in attn2_patches.items():
                        for attn_key, attn_tensor in attn_module.state_dict().items():
                            idx = idx_mapping.get(f"{k}-{attn_key}")
                            if idx is not None:
                                graph._state_tensor_tuple[idx].copy_(attn_tensor)
                                attn_module.set_attr(
                                    attn_key, graph._state_tensor_tuple[idx]
                                )
                                logger.debug(
                                    f"{graph._state_tensor_tuple[idx].shape} {attn_tensor.shape}"
                                )
            except Exception as e:
                logger.warning(f"Failed to apply patch after loading state dict: {e}")

        def process_state_dict_before_saving(state_dict: Dict):
            nonlocal self, args, kwargs, graph_file
            graph = self.get_graph()

            try:
                attn2_patches = (
                    state_dict.get("OneflowGraph_0", {})
                    .get("inputs_original", [])[1]
                    .get("transformer_options", {})
                    .get("patches_replace", {})
                    .get("attn2")
                )
                if attn2_patches:
                    id_index_map = {
                        id(t): i for i, t in enumerate(graph._state_tensor_tuple)
                    }

                    name_idx = {
                        f"{k}-{attn_k}": id_index_map[id(attn_t)]
                        for k, attn_m in attn2_patches.items()
                        for attn_k, attn_t in attn_m.state_dict().items()
                        if id(attn_t) in id_index_map
                    }

                    patch_file_name = f"{Path(graph_file).name}.patch"
                    new_file_path = Path(graph_file).with_name(patch_file_name)
                    with open(new_file_path, "wb") as fp:
                        pickle.dump(name_idx, fp)

                    del state_dict["OneflowGraph_0"]["inputs_original"][1][
                        "transformer_options"
                    ]["patches_replace"]["attn2"]

            except Exception as e:
                logger.debug(f"Failed to process state dict before saving: {e}")

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
                self.load_graph(graph_file, torch2oflow(graph_device))
                apply_patch_after_loading_state_dict()
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
