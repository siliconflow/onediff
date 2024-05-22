from typing import Optional
import oneflow as torch
import oneflow.nn as nn
import oneflow.nn.functional as F
from packaging import version
import importlib.metadata

from onediff.infer_compiler.backends.oneflow.transform import transform_mgr

transformed_diffusers = transform_mgr.transform_package("diffusers")

diffusers_0210_v = version.parse("0.21.0")
diffusers_0220_v = version.parse("0.22.0")
diffusers_version = version.parse(importlib.metadata.version("diffusers"))

if diffusers_version < diffusers_0210_v:

    class Upsample2D(transformed_diffusers.models.resnet.Upsample2D):
        def forward(
            self,
            hidden_states: torch.FloatTensor,
            output_size: Optional[int] = None,
            output_like: Optional[torch.FloatTensor] = None,
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
            if output_like is not None:
                hidden_states = F.interpolate_like(
                    hidden_states, like=output_like, mode="nearest"
                )
            elif output_size is None:
                hidden_states = F.interpolate(
                    hidden_states, scale_factor=2.0, mode="nearest"
                )
            else:
                hidden_states = F.interpolate(
                    hidden_states, size=output_size, mode="nearest"
                )

            # If the input is bfloat16, we cast back to bfloat16
            if dtype == torch.bfloat16:
                hidden_states = hidden_states.to(dtype)

            # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
            if self.use_conv:
                if self.name == "conv":
                    hidden_states = self.conv(hidden_states)
                else:
                    hidden_states = self.Conv2d_0(hidden_states)

            return hidden_states


else:

    class Upsample2D(transformed_diffusers.models.resnet.Upsample2D):
        def forward(
            self,
            hidden_states: torch.FloatTensor,
            output_size: Optional[int] = None,
            output_like: Optional[torch.FloatTensor] = None,
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
            if output_like is not None:
                hidden_states = F.interpolate_like(
                    hidden_states, like=output_like, mode="nearest"
                )
            elif output_size is None:
                hidden_states = F.interpolate(
                    hidden_states, scale_factor=2.0, mode="nearest"
                )
            else:
                hidden_states = F.interpolate(
                    hidden_states, size=output_size, mode="nearest"
                )

            # If the input is bfloat16, we cast back to bfloat16
            if dtype == torch.bfloat16:
                hidden_states = hidden_states.to(dtype)

            # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
            if self.use_conv:
                if self.name == "conv":
                    if diffusers_version < diffusers_0220_v:
                        if isinstance(
                            self.conv,
                            transformed_diffusers.models.lora.LoRACompatibleConv,
                        ):
                            hidden_states = self.conv(hidden_states, scale)
                        else:
                            hidden_states = self.conv(hidden_states)
                    else:
                        if (
                            isinstance(
                                self.conv,
                                transformed_diffusers.models.lora.LoRACompatibleConv,
                            )
                            and not transformed_diffusers.utils.USE_PEFT_BACKEND
                        ):
                            hidden_states = self.conv(hidden_states, scale)
                        else:
                            hidden_states = self.conv(hidden_states)
                else:
                    if diffusers_version < diffusers_0220_v:
                        if isinstance(
                            self.Conv2d_0,
                            transformed_diffusers.models.lora.LoRACompatibleConv,
                        ):
                            hidden_states = self.Conv2d_0(hidden_states, scale)
                        else:
                            hidden_states = self.Conv2d_0(hidden_states)
                    else:
                        if (
                            isinstance(
                                self.Conv2d_0,
                                transformed_diffusers.models.lora.LoRACompatibleConv,
                            )
                            and not transformed_diffusers.utils.USE_PEFT_BACKEND
                        ):
                            hidden_states = self.Conv2d_0(hidden_states, scale)
                        else:
                            hidden_states = self.Conv2d_0(hidden_states)

            return hidden_states
