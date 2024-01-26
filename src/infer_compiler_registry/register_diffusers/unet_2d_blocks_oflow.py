import oneflow as torch
import oneflow.nn.functional as F
from oneflow import nn

import importlib.metadata
from packaging import version
from typing import Any, Dict, List, Optional, Tuple, Union

from onediff.infer_compiler.transform import transform_mgr

transformed_diffusers = transform_mgr.transform_package("diffusers")
diffusers_version = version.parse(importlib.metadata.version("diffusers"))

LoRACompatibleConv = (
    transformed_diffusers.models.lora.LoRACompatibleConv
)

try:
    USE_PEFT_BACKEND = transformed_diffusers.utils.USE_PEFT_BACKEND
except Exception as e:
    USE_PEFT_BACKEND = False

if diffusers_version >= version.parse("0.25.00"):
    class Upsample2D(nn.Module):
        """A 2D upsampling layer with an optional convolution.
        """
        def forward(
            self,
            hidden_states: torch.FloatTensor,
            output_size: Optional[int] = None,
            scale: float = 1.0,
        ) -> torch.FloatTensor:
            assert hidden_states.shape[1] == self.channels

            if self.norm is not None:
                hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            if self.use_conv_transpose:
                return self.conv(hidden_states)

            # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
            # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
            # https://github.com/pytorch/pytorch/issues/86679
            dtype = hidden_states.dtype
            if dtype == torch.bfloat16:
                hidden_states = hidden_states.to(torch.float32)

            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                hidden_states = hidden_states.contiguous()

            # if `output_size` is passed we force the interpolation output
            # size and do not make use of `scale_factor=2`
            if self.interpolate:
                if output_size is None:
                    hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
                else:
                    # Rewritten for the switching of uncommon resolutions.
                    # hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")
                    hidden_states = F.interpolate_like(hidden_states, like=output_size, mode="nearest")

            # If the input is bfloat16, we cast back to bfloat16
            if dtype == torch.bfloat16:
                hidden_states = hidden_states.to(dtype)

            # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
            if self.use_conv:
                if self.name == "conv":
                    if isinstance(self.conv, LoRACompatibleConv) and not USE_PEFT_BACKEND:
                        hidden_states = self.conv(hidden_states, scale)
                    else:
                        hidden_states = self.conv(hidden_states)
                else:
                    if isinstance(self.Conv2d_0, LoRACompatibleConv) and not USE_PEFT_BACKEND:
                        hidden_states = self.Conv2d_0(hidden_states, scale)
                    else:
                        hidden_states = self.Conv2d_0(hidden_states)

            return hidden_states
else:
    class Upsample2D(nn.Module):
        """A 2D upsampling layer with an optional convolution.
        """
        def forward(
            self,
            hidden_states: torch.FloatTensor,
            output_size: Optional[int] = None,
            scale: float = 1.0,
        ) -> torch.FloatTensor:
            assert hidden_states.shape[1] == self.channels

            if self.use_conv_transpose:
                return self.conv(hidden_states)

            # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
            # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
            # https://github.com/pytorch/pytorch/issues/86679
            dtype = hidden_states.dtype
            if dtype == torch.bfloat16:
                hidden_states = hidden_states.to(torch.float32)

            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                hidden_states = hidden_states.contiguous()

            # if `output_size` is passed we force the interpolation output
            # size and do not make use of `scale_factor=2`
            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                # Rewritten for the switching of uncommon resolutions.
                # hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")
                hidden_states = F.interpolate_like(hidden_states, like=output_size, mode="nearest")

            # If the input is bfloat16, we cast back to bfloat16
            if dtype == torch.bfloat16:
                hidden_states = hidden_states.to(dtype)

            # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
            if self.use_conv:
                if self.name == "conv":
                    if isinstance(self.conv, LoRACompatibleConv) and not USE_PEFT_BACKEND:
                        hidden_states = self.conv(hidden_states, scale)
                    else:
                        hidden_states = self.conv(hidden_states)
                else:
                    if isinstance(self.Conv2d_0, LoRACompatibleConv) and not USE_PEFT_BACKEND:
                        hidden_states = self.Conv2d_0(hidden_states, scale)
                    else:
                        hidden_states = self.Conv2d_0(hidden_states)

            return hidden_states
