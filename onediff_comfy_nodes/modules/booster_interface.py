# import os
import dataclasses
import uuid
from abc import ABC, abstractmethod

# from functools import singledispatchmethod
# from typing import Optional

# import torch
# from comfy import model_management
# from comfy.controlnet import ControlLora, ControlNet
# from comfy.model_patcher import ModelPatcher
# from comfy.sd import VAE


class BoosterExecutor(ABC):
    """Interface for optimization."""

    @abstractmethod
    def execute(self, model, ckpt_name=None, **kwargs):
        """Apply the optimization strategy to the model."""
        pass


@dataclasses.dataclass
class BoosterSettings:
    tmp_cache_key: str = None


if __name__ == "__main__":
    print(BoosterSettings(str(uuid.uuid4())).tmp_cache_key)
    print(BoosterSettings(str(uuid.uuid4())).tmp_cache_key)
    print(BoosterSettings(str(uuid.uuid4())).tmp_cache_key)
