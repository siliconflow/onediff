import warnings
from typing import Dict, List, Optional, Union

import diffusers

import torch
from packaging import version

if version.parse(diffusers.__version__) >= version.parse("0.22.0"):
    from diffusers.utils.import_utils import is_peft_available

    if is_peft_available():
        import peft
else:
    is_peft_available = lambda: False

from onediff.infer_compiler.backends.oneflow.param_utils import (
    update_graph_related_tensor,
)

if version.parse(diffusers.__version__) <= version.parse("0.20.0"):
    from diffusers.loaders import PatchedLoraProjection
else:
    from diffusers.models.lora import PatchedLoraProjection

from onediff.infer_compiler.backends.oneflow.dual_module import DualModule


_adapter_layer_names = ()


def init_lora_infos(self: torch.nn.Module):
    self.r = {}
    self.lora_alpha = {}
    self.scaling = {}
    self.lora_A = {}
    self.lora_B = {}
    self.adapter_names = set()
    self.active_adapter_names = {}


def delete_lora_infos(
    self: Union[torch.nn.Linear, PatchedLoraProjection, torch.nn.Conv2d],
    adapter_names: Optional[List[str]] = None,
) -> None:

    if adapter_names is None:
        adapter_names = list(self.adapter_names.copy())
    curr_adapter_names = self.adapter_names.copy()
    for adapter_name in adapter_names:
        if adapter_name not in curr_adapter_names:
            continue
        self.r.pop(adapter_name)
        self.lora_alpha.pop(adapter_name)
        self.scaling.pop(adapter_name)
        self.lora_A.pop(adapter_name)
        self.lora_B.pop(adapter_name)
        self.adapter_names.remove(adapter_name)
        if adapter_name in self.active_adapter_names:
            self.active_adapter_names.pop(adapter_name)


def get_adapter_names(
    self: Union[torch.nn.Linear, PatchedLoraProjection, torch.nn.Conv2d],
):
    if not hasattr(self, "adapter_names"):
        result = "default_0"
    else:
        if not isinstance(self.adapter_names, set):
            adapter_names = set([self.adapter_names])
        else:
            adapter_names = self.adapter_names
        for i in range(0, 10000):
            result = f"default_{i}"
            if result not in adapter_names:
                break
    return result


def get_delta_weight(
    self: Union[torch.nn.Linear, PatchedLoraProjection, torch.nn.Conv2d],
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    if weight == 0:
        return torch.zeros_like(
            self.weight, dtype=self.weight.dtype, device=self.weight.device
        )

    if isinstance(self, (torch.nn.Linear, PatchedLoraProjection)):
        lora_weight = torch.bmm(w_up[None, :], w_down[None, :])[0]
    elif isinstance(self, torch.nn.Conv2d):
        lora_weight = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
        lora_weight = lora_weight.reshape((self.weight.shape))
    else:
        raise TypeError(
            f"[OneDiffX get_delta_weight] Expect type Linear or Conv2d, got {type(self)}"
        )
    if weight != 1.0:
        lora_weight *= weight
    return lora_weight


def offload_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    cur_device = tensor.device
    if cur_device == device:
        return tensor.clone()
    else:
        return tensor.to(device)


def _set_adapter(
    self: Union[torch.nn.Linear, PatchedLoraProjection, torch.nn.Conv2d],
    adapter_names: List[str],
    adapter_weights: Optional[Union[float, List[float]]] = None,
):
    if not isinstance(self, (torch.nn.Linear, torch.nn.Conv2d, PatchedLoraProjection)):
        raise TypeError(
            f"[OneDiffX _set_adapter] Expect type Linear or Conv2d, got {type(self)}"
        )
    if isinstance(self, PatchedLoraProjection):
        self = self.regular_linear_layer
    if not hasattr(self, "adapter_names"):
        return
    if adapter_weights is None:
        adapter_weights = [
            self.scaling[adapter]
            for adapter in adapter_names
            if adapter in self.scaling
        ]
    elif isinstance(adapter_weights, float):
        adapter_weights = [
            adapter_weights,
        ] * len(adapter_names)
    _unfuse_lora(self)

    dtype, device = self.weight.data.dtype, self.weight.data.device

    delta_weight = None
    for adapter, weight in zip(adapter_names, adapter_weights):
        if adapter not in self.adapter_names:
            continue

        self.active_adapter_names[adapter] = weight
        self.scaling[adapter] = weight * self.lora_alpha[adapter] / self.r[adapter]
        w_down = self.lora_A[adapter].float().to(device)
        w_up = self.lora_B[adapter].float().to(device)
        if delta_weight is None:
            delta_weight = get_delta_weight(self, w_up, w_down, self.scaling[adapter])
        else:
            delta_weight += get_delta_weight(self, w_up, w_down, self.scaling[adapter])

    if delta_weight is not None:
        fused_weight = self.weight.data.float() + delta_weight
        self.weight.data.copy_(fused_weight.to(device=device, dtype=dtype))
        update_graph_related_tensor(self)


def _delete_adapter(
    self: Union[torch.nn.Linear, PatchedLoraProjection, torch.nn.Conv2d],
    adapter_names: List[str],
    safe_delete: bool = True,
) -> None:
    if not isinstance(self, (torch.nn.Linear, torch.nn.Conv2d, PatchedLoraProjection)):
        raise TypeError(
            f"[OneDiffX _delete_adapter] Expect type Linear or Conv2d, got {type(self)}"
        )
    if isinstance(self, PatchedLoraProjection):
        self = self.regular_linear_layer
    if not hasattr(self, "adapter_names"):
        return
    if safe_delete:
        _set_adapter(self, list(set(self.adapter_names) - set(adapter_names)))
    else:
        _unfuse_lora(self, adapter_names=adapter_names)
    delete_lora_infos(self, adapter_names)


def _load_lora_and_optionally_fuse(
    self: Union[torch.nn.Linear, PatchedLoraProjection, torch.nn.Conv2d],
    state_dict: Dict[str, torch.Tensor],
    lora_scale: Optional[float] = None,
    alpha: Optional[float] = None,
    rank: Optional[float] = None,
    *,
    adapter_name: Optional[str] = None,
    fuse: bool = True,
    prefix: str = "lora",
    offload_device: str = "cpu",
) -> None:
    r"""
    This will fuse the LoRA weights in `state_dict` into Linear or Conv2d module.

    Parameters:
        self (Union[torch.nn.Linear, PatchedLoraProjection, torch.nn.Conv2d]):
            Model layer to be fused, must be Linear or PatchedLoraProjection or Conv2d.
        state_dict (Dict[str, torch.Tensor]):
            Dictionary containing LoRA weight.
        lora_scale (float, optional):
            Scaling factor for LoRA weights. Default is 1.0.
        alpha (float, optional):
            Alpha parameter of LoRA weights. Default is None.
        rank (float, optional):
            Rank of LoRA weights. Default is None.
        adapter_names (str, optional):
            The name of the adapter to be set for the LoRA. If not provided, a default name will be generated.
        fuse (bool, optional):
            Determines whether to fuse the LoRA weights into the module. If False, the LoRA weights are merely loaded without fusion.
            Defaults to True.
        prefix (str, optional):
            Prefix for up and down weight keys in the LoRA weight dictionary. Default is "lora".
        offload_device (str, optional):
            Offload Device for backuping weight, can be "cpu" or "cuda". Default is "cpu".
    """
    if not isinstance(self, (torch.nn.Linear, PatchedLoraProjection, torch.nn.Conv2d)):
        if is_peft_available() and isinstance(
            self, (peft.tuners.lora.layer.Linear, peft.tuners.lora.layer.Conv2d)
        ):
            self = self.base_layer
        else:
            raise TypeError(
                f"[OneDiffX fuse_lora] Only Linear and Conv2d can fuse lora, but got type {type(self)}"
            )

    if isinstance(self, DualModule):
        self = self._torch_module
    if isinstance(self, PatchedLoraProjection):
        self = self.regular_linear_layer

    if not hasattr(self, "adapter_names"):
        init_lora_infos(self)

    dtype, device = self.weight.data.dtype, self.weight.data.device

    down_key = prefix + ".down.weight"
    up_key = prefix + ".up.weight"

    w_down = state_dict[down_key].float().to(device)
    w_up = state_dict[up_key].float().to(device)

    adapter_name = adapter_name if adapter_name is not None else get_adapter_names(self)

    if alpha is None:
        alpha = rank

    if lora_scale is None:
        lora_scale = 1.0

    self.scaling[adapter_name] = lora_scale * alpha / rank
    self.r[adapter_name] = rank
    self.lora_alpha[adapter_name] = alpha
    self.lora_A[adapter_name] = offload_tensor(w_down, offload_device)
    self.lora_B[adapter_name] = offload_tensor(w_up, offload_device)
    self.adapter_names.add(adapter_name)

    if fuse:
        if len(self.active_adapter_names) != 0 and dtype in [
            torch.half,
            torch.bfloat16,
        ]:
            warnings.warn(
                "There is already merged LoRA adapters. "
                "If you need to load another LoRA, please unmerge the existing LoRA adapters. "
                "Otherwise, it may lead to accuracy issues, impacting the quality of the generated images."
            )
        self.active_adapter_names[adapter_name] = lora_scale
        lora_weight = get_delta_weight(self, w_up, w_down, self.scaling[adapter_name])
        fused_weight = self.weight.data.float() + lora_weight
        self.weight.data.copy_(fused_weight.to(device=device, dtype=dtype))
        update_graph_related_tensor(self)


def _unfuse_lora(
    self: Union[torch.nn.Linear, PatchedLoraProjection, torch.nn.Conv2d],
    adapter_names: Optional[Union[str, List[str]]] = None,
) -> None:
    assert isinstance(self, (torch.nn.Linear, PatchedLoraProjection, torch.nn.Conv2d))
    if not hasattr(self, "adapter_names"):
        return
    if isinstance(self, DualModule):
        self = self._torch_module
    if isinstance(self, PatchedLoraProjection):
        self = self.regular_linear_layer

    fused_weight = self.weight.data
    dtype, device = fused_weight.dtype, fused_weight.device

    delta_weight = None
    if adapter_names is None:
        adapter_names = self.active_adapter_names.copy()

    for name in adapter_names:
        if name not in self.active_adapter_names:
            continue
        w_down = self.lora_A[name].to(device=device).float()
        w_up = self.lora_B[name].to(device).float()
        if delta_weight is None:
            delta_weight = get_delta_weight(self, w_up, w_down, self.scaling[name]).to(
                dtype=dtype, device=device
            )
        else:
            delta_weight += get_delta_weight(self, w_up, w_down, self.scaling[name]).to(
                dtype=dtype, device=device
            )
        self.active_adapter_names.pop(name)

    if delta_weight is not None:
        self.weight.data -= delta_weight
        update_graph_related_tensor(self)


# the code is referenced from https://github.com/huggingface/diffusers/blob/ce9825b56bd8a6849e68b9590022e935400659e6/src/diffusers/loaders/lora_conversion_utils.py#L24
@classmethod
def _maybe_map_sgm_blocks_to_diffusers(
    cls, state_dict, unet_config, delimiter="_", block_slice_pos=5
):
    # 1. get all state_dict_keys
    all_keys = list(state_dict.keys())
    sgm_patterns = ["input_blocks", "middle_block", "output_blocks"]

    # 2. check if needs remapping, if not return original dict
    is_in_sgm_format = False
    for key in all_keys:
        if any(p in key for p in sgm_patterns):
            is_in_sgm_format = True
            break

    if not is_in_sgm_format:
        return state_dict
    # 3. Else remap from SGM patterns
    new_state_dict = {}
    inner_block_map = ["resnets", "attentions", "upsamplers"]

    # Retrieves # of down, mid and up blocks
    input_block_ids, middle_block_ids, output_block_ids = set(), set(), set()

    for layer in all_keys:
        if "text" in layer:
            new_state_dict[layer] = state_dict.pop(layer)
        else:
            layer_id = int(layer.split(delimiter)[:block_slice_pos][-1])
            if sgm_patterns[0] in layer:
                input_block_ids.add(layer_id)
            elif sgm_patterns[1] in layer:
                middle_block_ids.add(layer_id)
            elif sgm_patterns[2] in layer:
                output_block_ids.add(layer_id)
            else:
                raise ValueError(
                    f"Checkpoint not supported because layer {layer} not supported."
                )

    input_blocks = {
        layer_id: [
            key for key in state_dict if f"input_blocks{delimiter}{layer_id}" in key
        ]
        for layer_id in input_block_ids
    }
    middle_blocks = {
        layer_id: [
            key for key in state_dict if f"middle_block{delimiter}{layer_id}" in key
        ]
        for layer_id in middle_block_ids
    }
    output_blocks = {
        layer_id: [
            key for key in state_dict if f"output_blocks{delimiter}{layer_id}" in key
        ]
        for layer_id in output_block_ids
    }

    # Rename keys accordingly
    for i in input_block_ids:
        block_id = (i - 1) // (unet_config.layers_per_block + 1)
        layer_in_block_id = (i - 1) % (unet_config.layers_per_block + 1)

        for key in input_blocks[i]:
            inner_block_id = int(key.split(delimiter)[block_slice_pos])
            inner_block_key = (
                inner_block_map[inner_block_id] if "op" not in key else "downsamplers"
            )
            inner_layers_in_block = str(layer_in_block_id) if "op" not in key else "0"
            new_key = delimiter.join(
                key.split(delimiter)[: block_slice_pos - 1]
                + [str(block_id), inner_block_key, inner_layers_in_block]
                + key.split(delimiter)[block_slice_pos + 1 :]
            )
            new_state_dict[new_key] = state_dict.pop(key)

    for i in middle_block_ids:
        key_part = None
        if i == 0:
            key_part = [inner_block_map[0], "0"]
        elif i == 1:
            key_part = [inner_block_map[1], "0"]
        elif i == 2:
            key_part = [inner_block_map[0], "1"]
        else:
            raise ValueError(f"Invalid middle block id {i}.")

        for key in middle_blocks[i]:
            new_key = delimiter.join(
                key.split(delimiter)[: block_slice_pos - 1]
                + key_part
                + key.split(delimiter)[block_slice_pos:]
            )
            new_state_dict[new_key] = state_dict.pop(key)

    for i in output_block_ids:
        block_id = i // (unet_config.layers_per_block + 1)
        layer_in_block_id = i % (unet_config.layers_per_block + 1)

        for key in output_blocks[i]:
            inner_block_id = int(key.split(delimiter)[block_slice_pos])
            inner_block_key = inner_block_map[inner_block_id]
            inner_layers_in_block = (
                str(layer_in_block_id) if inner_block_id < 2 else "0"
            )
            new_key = delimiter.join(
                key.split(delimiter)[: block_slice_pos - 1]
                + [str(block_id), inner_block_key, inner_layers_in_block]
                + key.split(delimiter)[block_slice_pos + 1 :]
            )
            new_state_dict[new_key] = state_dict.pop(key)

    if len(state_dict) > 0:
        raise ValueError("At this point all state dict entries have to be converted.")

    return new_state_dict
