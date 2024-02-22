from typing import Dict, Union, List
from packaging import version
from collections import OrderedDict

import torch
import diffusers
from onediff.infer_compiler.with_oneflow_compile import DualModule

if version.parse(diffusers.__version__) <= version.parse("0.20.0"):
    from diffusers.loaders import PatchedLoraProjection
else:
    from diffusers.models.lora import PatchedLoraProjection

_adapter_layer_names = ()


def init_lora_infos(self: torch.nn.Module):
    self.r = {}
    self.lora_alpha = {}
    self.scaling = {}
    self.lora_A = {}
    self.lora_B = {}
    self.adapter_names = set()
    self.active_adapter_names = set()


def get_adapter_names(self):
    if not hasattr(self, "adapter_names"):
        result = "default_0"
    else:
        if not isinstance(self.adapter_names, set):
            adapter_names = set([self.adapter_names])
        else:
            adapter_names = self.adapter_names
        for i in range(0, 100):
            result = f"default_{i}"
            if result not in adapter_names:
                break
    return result


def get_delta_weight(
    self: Union[torch.nn.Linear, PatchedLoraProjection, torch.nn.Conv2d],
    w_up: torch.Tensor,
    w_down: torch.Tensor,
):
    if isinstance(self, (torch.nn.Linear, PatchedLoraProjection)):
        lora_weight = torch.bmm(w_up[None, :], w_down[None, :])[0]
    elif isinstance(self, torch.nn.Conv2d):
        lora_weight = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
        lora_weight = lora_weight.reshape((self.weight.shape))
    else:
        raise TypeError
    return lora_weight


def offload_tensor(tensor, device):
    cur_device = tensor.device
    if cur_device == device:
        return tensor.clone()
    else:
        return tensor.to(device)


def set_adapter(self, adapter_names, adapter_weights):
    if not isinstance(self, (torch.nn.Linear, torch.nn.Conv2d)):
        raise
    if not hasattr(self, "adapter_names"):
        return

    if isinstance(self, torch.nn.Linear):
        _linear_unfuse_lora(self)
    else:
        _conv_unfuse_lora(self)

    dtype, device = self.weight.data.dtype, self.weight.data.device

    delta_weight = None
    for adapter, weight in zip(adapter_names, adapter_weights):
        if adapter not in self.adapter_names:
            continue

        self.active_adaptive_names.add(adapter_names)
        w_down = self.lora_A[adapter].float().to(device)
        w_up = self.lora_B[adapter].float().to(device)
        if delta_weight is None:
            delta_weight = torch.bmm(w_up[None, :], w_down[None, :])[0] * (
                weight / self.scaling[adapter]
            )
        else:
            delta_weight += torch.bmm(w_up[None, :], w_down[None, :])[0] * (
                weight / self.scaling[adapter]
            )
        self.weight.data += delta_weight.to(device=device, dtype=dtype)


def fuse_lora(
    self: Union[torch.nn.Linear, PatchedLoraProjection, torch.nn.Conv2d],
    state_dict: Dict[str, torch.Tensor] = None,
    lora_scale: float = 1.0,
    alpha: float = None,
    rank: float = None,
    *,
    adapter_names=None,
    fuse=True,
    prefix="lora",
    offload_device="cpu",
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
        prefix (str, optional):
            Prefix for up and down weight keys in the LoRA weight dictionary. Default is "lora".
        offload_device (str, optional):
            Offload Device for backuping weight, can be "cpu" or "cuda". Default is "cpu".
    """
    assert isinstance(self, (torch.nn.Linear, PatchedLoraProjection, torch.nn.Conv2d))
    
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

    if alpha is not None:
        w_up = w_up * (alpha / rank * lora_scale)

    adapter_names = (
        adapter_names if adapter_names is not None else get_adapter_names(self)
    )

    self.r[adapter_names] = rank
    self.lora_alpha[adapter_names] = alpha
    self.scaling[adapter_names] = lora_scale
    self.lora_A[adapter_names] = offload_tensor(w_down, offload_device)
    self.lora_B[adapter_names] = offload_tensor(w_up, offload_device)
    self.adapter_names.add(adapter_names)
    self.active_adapter_names.add(adapter_names)

    if fuse:
        lora_weight = get_delta_weight(self, w_up, w_down)
        fused_weight = self.weight.data.float() + lora_weight
        self.weight.data.copy_(fused_weight.to(device=device, dtype=dtype))


def _linear_unfuse_lora(
    self: Union[torch.nn.Linear, PatchedLoraProjection],
    adapter_names: Union[str, List[str]] = None,
):
    assert isinstance(self, (torch.nn.Linear, PatchedLoraProjection))
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
            delta_weight = (torch.bmm(w_up[None, :], w_down[None, :])[0]).to(
                device=device, dtype=dtype
            )
        else:
            delta_weight += (torch.bmm(w_up[None, :], w_down[None, :])[0]).to(
                device=device, dtype=dtype
            )
        self.active_adapter_names.remove(name)

    if delta_weight is not None:
        self.weight.data -= delta_weight


def _conv_unfuse_lora(
    self: torch.nn.Conv2d, adapter_names: Union[str, List[str]] = None
):
    assert isinstance(self, torch.nn.Conv2d)
    if not hasattr(self, "adapter_names"):
        return

    fused_weight = self.weight.data
    dtype, device = fused_weight.dtype, fused_weight.device

    delta_weight_sum = None
    if adapter_names is None:
        adapter_names = self.active_adapter_names.copy()

    for name in adapter_names:
        if name not in self.active_adapter_names:
            continue
        w_down = self.lora_A[name].to(device=device).float()
        w_up = self.lora_B[name].to(device).float()
        delta_weight = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
        if delta_weight_sum is None:
            delta_weight_sum = delta_weight.reshape((fused_weight.shape)).to(
                device=device, dtype=dtype
            )
        else:
            delta_weight_sum += delta_weight.reshape((fused_weight.shape)).to(
                device=device, dtype=dtype
            )
        self.active_adapter_names.remove(name)

    if delta_weight_sum is not None:
        self.weight.data -= delta_weight_sum
