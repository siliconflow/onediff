# ONEDIFF_TORCH_TO_ONEF_CLASS_MAP = { PYTORCH_MODEL_CLASS: ONEFLOW_MODEL_CLASS }
# ONEDIFF_CUSTOM_TORCH2OF_FUNC_TYPE_MAP = { Function :  TYPE }

from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from .attention_processor_1f import Attention as Attention1f
from .attention_processor_1f import AttnProcessor as AttnProcessor1f

ONEDIFF_TORCH_TO_ONEF_CLASS_MAP = {
    Attention: Attention1f,
    AttnProcessor2_0: AttnProcessor1f,
}
