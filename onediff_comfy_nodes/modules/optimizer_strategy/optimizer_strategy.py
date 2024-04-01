from abc import ABC, abstractmethod

from comfy.model_patcher import ModelPatcher


class OptimizerStrategy(ABC):
    """Interface for optimization strategies."""

    @abstractmethod
    def apply(self, model: ModelPatcher):
        """Apply the optimization strategy to the model."""
        pass

