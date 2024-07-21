from typing import Any

import torch


class DeployableModule(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError()
