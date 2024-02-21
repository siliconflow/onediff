from typing import Dict, Union
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


def get_adapter_name(self):
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


def offload_tensor(tensor, device):
    cur_device = tensor.device
    if cur_device == device:
        return tensor.clone()
    else:
        return tensor.to(device)


def linear_fuse_lora(
    self: Union[torch.nn.Linear, PatchedLoraProjection],
    state_dict: Dict[str, torch.Tensor],
    lora_scale: float = 1.0,
    alpha: float = None,
    rank: float = None,
    *,
    adapter_name=None,
    fuse=True,
    prefix="lora",
    offload_device="cpu",
):
    r"""
    This will fuse the LoRA weights in `state_dict` into Linear module.

    Parameters:
        self (Union[torch.nn.Linear, PatchedLoraProjection]):
            Model layer to be fused, must be Linear or PatchedLoraProjection.
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
    assert isinstance(self, (torch.nn.Linear, PatchedLoraProjection))
    if isinstance(self, DualModule):
        self = self._torch_module
    if isinstance(self, PatchedLoraProjection):
        self = self.regular_linear_layer

    dtype, device = self.weight.data.dtype, self.weight.data.device
    down_key = prefix + ".down.weight"
    up_key = prefix + ".up.weight"

    w_down = state_dict[down_key].float().to(device)
    w_up = state_dict[up_key].float().to(device)

    if alpha is not None:
        w_up = w_up * (alpha / rank * lora_scale)

    adapter_name = adapter_name if adapter_name is not None else get_adapter_name(self)

    if not hasattr(self, "adapter_names"):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_A = {}
        self.lora_B = {}
        self.adapter_names = set()
        self.active_adapter_names = set()

    self.r[adapter_name] = rank
    self.lora_alpha[adapter_name] = alpha
    self.scaling[adapter_name] = lora_scale
    self.lora_A[adapter_name] = offload_tensor(w_down, offload_device)
    self.lora_B[adapter_name] = offload_tensor(w_up, offload_device)
    self.adapter_names.add(adapter_name)
    self.active_adapter_names.add(adapter_name)

    if fuse:
        lora_weight = torch.bmm(w_up[None, :], w_down[None, :])[0]
        fused_weight = self.weight.data.float() + lora_weight
        self.weight.data.copy_(fused_weight.to(device=device, dtype=dtype))


def _linear_unfuse_lora(self: Union[torch.nn.Linear, PatchedLoraProjection]):
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
    for name in self.adapter_names:
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
    self.adapter_names.clear()

    if delta_weight is not None:
        self.weight.data -= delta_weight


def conv_fuse_lora(
    self: torch.nn.Conv2d,
    state_dict: Dict[str, torch.Tensor],
    lora_scale: float = 1.0,
    alpha: float = None,
    rank: float = None,
    *,
    adapter_name=None,
    fuse=True,
    prefix="lora",
    offload_device="cpu",
) -> None:
    r"""
    This will fuse the LoRA weights in `state_dict` into Conv2d module.

    Parameters:
        self (torch.nn.Conv2d):
            Model layer to be fused, must be torch.nn.Conv2d.
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
    assert isinstance(self, torch.nn.Conv2d)
    if isinstance(self, DualModule):
        self = self._torch_module
    dtype, device = self.weight.data.dtype, self.weight.data.device

    down_key = prefix + ".down.weight"
    up_key = prefix + ".up.weight"
    w_down = state_dict[down_key].float().to(device)
    w_up = state_dict[up_key].float().to(device)

    if alpha is not None:
        w_up = w_up * (alpha / rank * lora_scale)

    adapter_name = adapter_name if adapter_name is not None else get_adapter_name(self)

    if not hasattr(self, "adapter_names"):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_A = {}
        self.lora_B = {}
        self.adapter_names = set()

    self.r[adapter_name] = rank
    self.lora_alpha[adapter_name] = alpha
    self.scaling[adapter_name] = lora_scale
    self.lora_A[adapter_name] = offload_tensor(w_down, offload_device)
    self.lora_B[adapter_name] = offload_tensor(w_up, offload_device)
    self.adapter_names.add(adapter_name)

    if fuse:
        lora_weight = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
        lora_weight = lora_weight.reshape((self.weight.shape))

        fused_weight = self.weight.data.float() + lora_weight
        self.weight.data.copy_(fused_weight.to(device=device, dtype=dtype))


def _conv_unfuse_lora(self: torch.nn.Conv2d):
    assert isinstance(self, torch.nn.Conv2d)
    if not hasattr(self, "adapter_names"):
        return

    fused_weight = self.weight.data
    dtype, device = fused_weight.dtype, fused_weight.device

    delta_weight_sum = None
    for name in self.adapter_names:
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

    self.adapter_names.clear()

    if delta_weight_sum is not None:
        self.weight.data -= delta_weight_sum
