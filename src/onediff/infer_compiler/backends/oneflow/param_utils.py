import re
import types

import torch
import oneflow as flow  # usort: skip
from typing import Any, Dict, List, Union

from onediff.utils import logger


def parse_device(args: List[Any], kwargs: Dict[str, Any]):
    if "device" in kwargs:
        return kwargs["device"]
    for x in args:
        if isinstance(x, (flow.device, torch.device)):
            return x
        if x in ["cpu", "cuda"]:
            return x
    return None


def check_device(current_device, target_device) -> bool:
    def _convert(device):
        assert isinstance(device, (str, torch.device, flow.device))
        if isinstance(device, torch.device):
            index = device.index if device.index is not None else 0
            return flow.device(device.type, index)
        if isinstance(device, str):
            return flow.device(device)
        return device

    return _convert(current_device) == _convert(target_device)


# hooks and helper functions for constant folding conv weights

STATE_UPDATED_ATTR = "_onediff_state_updated"
CONSTANT_FOLDING_INFO_ATTR = "_onediff_constant_folding_info"
GRAPH_RELATED_TENSOR_ATTR = "_onediff_graph_related_tensor"


def init_state_update_attr(module: torch.nn.Module):
    from onediff.infer_compiler import DeployableModule

    if isinstance(module, DeployableModule):
        module = module._torch_module
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"module must be a torch.nn.Module, got {type(module)}")
    setattr(module, STATE_UPDATED_ATTR, False)


def set_constant_folded_conv_attr(
    deployable_module, constant_folding_info: Dict[str, flow.Tensor] = None
) -> None:
    from onediff.infer_compiler import DeployableModule

    if not isinstance(deployable_module, DeployableModule):
        raise TypeError(
            f"deployable_model must be a DeployableModule, got {type(deployable_module)}"
        )

    constant_folding_info = constant_folding_info or get_constant_folding_info(
        deployable_module
    )
    if constant_folding_info is None:
        return

    torch_module: torch.nn.Module = deployable_module._torch_module
    for submodule in torch_module.modules():
        if isinstance(submodule, torch.nn.Conv2d) and hasattr(
            submodule, GRAPH_RELATED_TENSOR_ATTR
        ):
            delattr(submodule, GRAPH_RELATED_TENSOR_ATTR)

    for weight_name, weight_tensor in constant_folding_info.items():
        submodule = deployable_module._torch_module.get_submodule(
            removesuffix(weight_name, ".weight")
        )
        object.__setattr__(submodule, GRAPH_RELATED_TENSOR_ATTR, weight_tensor)


def generate_constant_folding_info(
    deployable_module, torch_module: torch.nn.Module = None
) -> Dict[str, flow.Tensor]:
    removeprefix = lambda ss, prefix: ss[len(prefix) :] if ss.startswith(prefix) else ss

    # convert str like 'variable_transpose_model.input_blocks.10.0.in_layers.2.weight_239'
    # to 'input_blocks.10.0.in_layers.2.weight'
    def convert_var_name(s: str, prefix="variable_transpose_"):
        s = removeprefix(s, prefix)
        s = re.sub(r"_[0-9]+$", "", s)
        s = removeprefix(s, "model.")
        return s

    from onediff.infer_compiler import DeployableModule

    if not isinstance(deployable_module, DeployableModule):
        raise TypeError(
            f"deployable_model must be a DeployableModule, got {type(deployable_module)}"
        )
    if torch_module is None:
        torch_module = deployable_module._torch_module

    graph = deployable_module._deployable_module_dpl_graph
    if graph is None:
        raise RuntimeError(f"The graph of deployable_module is not built yet")

    result = {
        convert_var_name(k): v
        for k, v in zip(*graph._c_nn_graph.get_runtime_var_states())
        if k.startswith("variable_transpose_") and v.ndim == 4
    }

    setattr(deployable_module, CONSTANT_FOLDING_INFO_ATTR, result)

    set_constant_folded_conv_attr(deployable_module, result)

    def make_custom_copy_(module):
        def custom_copy_(self, src, non_blocking=False):
            torch.Tensor.copy_(self, src, non_blocking)
            # Update graph related tensors
            update_graph_related_tensor(module)

        return custom_copy_

    from onediff.torch_utils.module_operations import get_sub_module

    torch_model: torch.nn.Module = deployable_module._torch_module
    for k in result.keys():
        module = get_sub_module(torch_model, removesuffix(k, ".weight"))
        module.weight.copy_ = types.MethodType(make_custom_copy_(module), module.weight)


def update_graph_with_constant_folding_info(
    module: torch.nn.Module, info: Dict[str, flow.Tensor] = None
) -> None:
    from onediff.infer_compiler import DeployableModule

    if isinstance(module, DeployableModule):
        if info is None:
            info = get_constant_folding_info(module)
        module = module._torch_module
    if info is None:
        return

    for k in info:
        orig_tensor = module.get_parameter(k)
        target_tensor = info.get(k, None)
        if target_tensor is None:
            raise RuntimeError(f"Can't find tensor named {k} in graph")
        target_tensor.copy_(
            flow.utils.tensor.from_torch(orig_tensor.permute(0, 2, 3, 1))
        )


def update_graph_related_tensor(module: torch.nn.Conv2d) -> None:
    if not isinstance(module, torch.nn.Conv2d):
        return
    target_tensor = getattr(module, GRAPH_RELATED_TENSOR_ATTR, None)
    if target_tensor is None:
        return
    target_tensor.copy_(
        flow.utils.tensor.from_torch(module.weight.data.permute(0, 2, 3, 1))
    )


def get_constant_folding_info(module) -> Union[Dict[str, flow.Tensor], None]:
    from onediff.infer_compiler import DeployableModule

    if not isinstance(module, DeployableModule):
        raise TypeError(f"module must be a DeployableModule, got {type(module)}")
    return getattr(module, CONSTANT_FOLDING_INFO_ATTR, None)


def state_update_hook(module, incompatible_keys):
    if not hasattr(module, STATE_UPDATED_ATTR):
        return
    logger.info(f"load_state_dict called, set {STATE_UPDATED_ATTR} to True")
    setattr(module, STATE_UPDATED_ATTR, True)


def forward_generate_constant_folding_info_hook(module, args, output):
    if module._deployable_module_dpl_graph is None:
        return

    if getattr(module, CONSTANT_FOLDING_INFO_ATTR, None) is not None:
        return

    logger.info(f"generate constant folding info")
    generate_constant_folding_info(module)


def forward_pre_check_and_update_state_hook(module, args):
    if module._deployable_module_dpl_graph is None:
        return

    if not getattr(module._torch_module, STATE_UPDATED_ATTR, False):
        return

    constant_folding_info = getattr(module, CONSTANT_FOLDING_INFO_ATTR, None)
    if constant_folding_info is None:
        return

    logger.info(f"state_dict updated, modify the related weight in graph")
    update_graph_with_constant_folding_info(module, constant_folding_info)
    setattr(module._torch_module, STATE_UPDATED_ATTR, False)


def removesuffix(s: str, suffix: str) -> str:
    if s.endswith(suffix):
        return s[: len(s) - len(suffix)]
    else:
        return s
