from abc import ABC, abstractmethod
from .log_utils import LOGGER

try:
    import diffusers
    from diffusers.models.attention_processor import Attention
except ImportError:
    diffusers = None
    LOGGER.warning("diffusers not found, some features will be disabled.")

_IS_DIFFUSERS_AVAILABLE = diffusers is not None


class InstanceChecker(ABC):
    @abstractmethod
    def is_attention_instance(self, instance):
        pass


class DiffusersChecker(InstanceChecker):
    def is_attention_instance(self, instance):
        if not _IS_DIFFUSERS_AVAILABLE:
            return False
        return isinstance(instance, Attention)


diffusers_checker = DiffusersChecker()
