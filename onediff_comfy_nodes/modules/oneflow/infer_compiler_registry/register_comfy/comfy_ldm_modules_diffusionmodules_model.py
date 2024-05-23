from typing import Any, Optional

import comfy
import oneflow as torch
import oneflow.nn as nn
import oneflow.nn.functional as F
from onediff.infer_compiler.backends.oneflow.transform import proxy_class


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )
