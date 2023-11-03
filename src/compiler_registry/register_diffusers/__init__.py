from onediff.infer_compiler.registry import register



from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from .attention_processor_1f import Attention as Attention1f
from .attention_processor_1f import AttnProcessor as AttnProcessor1f

torch2of_class_map = {
    Attention: Attention1f,
    AttnProcessor2_0: AttnProcessor1f,
}

register(package_names=["diffusers"], torch2of_class_map=torch2of_class_map)

