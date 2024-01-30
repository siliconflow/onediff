from typing import Dict, Union
from packaging import version

import torch
import diffusers
from onediff.infer_compiler.with_oneflow_compile import DualModule

if version.parse(diffusers.__version__) <= version.parse("0.20.0"):
    from diffusers.loaders import PatchedLoraProjection
else:
    from diffusers.models.lora import PatchedLoraProjection

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
    prefix="lora",
    offload_device="cpu",
    offload_weight="lora",
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
        offload_weight (str, optional):
            Which weights to be offloaded, can be "lora" or "weight". Default is "lora".
            If set to "weight", the weight of Linear rather than LoRA will be saved for unfusing.
    """
    assert isinstance(self, (torch.nn.Linear, PatchedLoraProjection))
    if isinstance(self, DualModule):
        self = self._torch_module
    if isinstance(self, PatchedLoraProjection):
        self = self.regular_linear_layer

    _linear_unfuse_lora(self)
    dtype, device = self.weight.data.dtype, self.weight.data.device
    down_key = prefix + ".down.weight"
    up_key = prefix + ".up.weight"

    w_down = state_dict[down_key].float().to(device)
    w_up = state_dict[up_key].float().to(device)

    if alpha is not None:
        w_up = w_up * (alpha / rank * lora_scale)

    if offload_weight == "lora":
        self.register_buffer("_lora_up", offload_tensor(w_up, offload_device))
        self.register_buffer(
            "_lora_down", offload_tensor(state_dict[down_key], offload_device)
        )
        self._lora_scale = lora_scale

    elif offload_weight == "weight":
        self.register_buffer(
            "_lora_orig_weight", offload_tensor(self.weight.data, offload_device)
        )

    else:
        raise ValueError(
            f"[OneDiff linear_fuse_lora] Invalid offload weight: {offload_weight}"
        )

    lora_weight = torch.bmm(w_up[None, :], w_down[None, :])[0]
    fused_weight = self.weight.data.float() + lora_weight
    self.weight.data.copy_(fused_weight.to(device=device, dtype=dtype))


def _linear_unfuse_lora(self: Union[torch.nn.Linear, PatchedLoraProjection]):
    assert isinstance(self, (torch.nn.Linear, PatchedLoraProjection))
    if isinstance(self, DualModule):
        self = self._torch_module
    if isinstance(self, PatchedLoraProjection):
        self = self.regular_linear_layer

    fused_weight = self.weight.data
    dtype, device = fused_weight.dtype, fused_weight.device

    if (
        "_lora_orig_weight" in self._buffers
        and self.get_buffer("_lora_orig_weight") is not None
    ):
        unfused_weight = self._lora_orig_weight
        self._lora_orig_weight = None

    elif "_lora_up" in self._buffers and self.get_buffer("_lora_up") is not None:
        w_up = self.get_buffer("_lora_up").to(device=device).float()
        w_down = self.get_buffer("_lora_down").to(device).float()

        unfused_weight = self.weight.data.float() - (
            torch.bmm(w_up[None, :], w_down[None, :])[0]
        )
        self._lora_up = None
        self._lora_down = None
        self._lora_scale = None

    else:
        return

    self.weight.data.copy_(unfused_weight.to(device=device, dtype=dtype))


def conv_fuse_lora(
    self: torch.nn.Conv2d,
    state_dict: Dict[str, torch.Tensor],
    lora_scale: float = 1.0,
    alpha: float = None,
    rank: float = None,
    *,
    prefix="lora",
    offload_device="cpu",
    offload_weight="lora",
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
        offload_weight (str, optional):
            Which weights to be offloaded, can be "lora" or "weight". Default is "lora".
            If set to "weight", the weight of Conv2d rather than LoRA will be saved for unfusing.
    """
    assert isinstance(self, torch.nn.Conv2d)
    if isinstance(self, DualModule):
        self = self._torch_module
    _conv_unfuse_lora(self)
    dtype, device = self.weight.data.dtype, self.weight.data.device

    down_key = prefix + ".down.weight"
    up_key = prefix + ".up.weight"
    w_down = state_dict[down_key].float().to(device)
    w_up = state_dict[up_key].float().to(device)

    if alpha is not None:
        w_up = w_up * (alpha / rank * lora_scale)

    if offload_weight == "lora":
        self.register_buffer("_lora_up", offload_tensor(w_up, offload_device))
        self.register_buffer(
            "_lora_down", offload_tensor(state_dict[down_key], offload_device)
        )
        self._lora_scale = lora_scale
    elif offload_weight == "weight":
        self.register_buffer(
            "_lora_orig_weight", offload_tensor(self.weight.data, offload_device)
        )
    else:
        raise ValueError(
            f"[OneDiff conv_fuse_lora] Invalid offload weight: {offload_weight}"
        )

    lora_weight = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
    lora_weight = lora_weight.reshape((self.weight.shape))

    fused_weight = self.weight.data.float() + lora_weight
    self.weight.data.copy_(fused_weight.to(device=device, dtype=dtype))


def _conv_unfuse_lora(self: torch.nn.Conv2d):
    assert isinstance(self, torch.nn.Conv2d)

    fused_weight = self.weight.data
    dtype, device = fused_weight.dtype, fused_weight.device

    if (
        "_lora_orig_weight" in self._buffers
        and self.get_buffer("_lora_orig_weight") is not None
    ):
        unfused_weight = self._lora_orig_weight
        self._lora_orig_weight = None

    elif "_lora_up" in self._buffers and self.get_buffer("_lora_up") is not None:
        w_up = self._lora_up.to(device=device).float()
        w_down = self._lora_down.to(device).float()

        fusion = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
        fusion = fusion.reshape((fused_weight.shape))
        unfused_weight = fused_weight.float() - fusion

        self._lora_up = None
        self._lora_down = None
        self._lora_scale = None

    else:
        return

    self.weight.data.copy_(unfused_weight.to(device=device, dtype=dtype))
