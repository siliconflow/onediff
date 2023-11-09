from onediff.infer_compiler.registry import register



from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from .attention_processor_oflow import Attention as AttentionOflow
from .attention_processor_oflow import AttnProcessor as AttnProcessorOflow

torch2oflow_class_map = {
    Attention: AttentionOflow,
    AttnProcessor2_0: AttnProcessorOflow,
}

register(package_names=["diffusers"], torch2oflow_class_map=torch2oflow_class_map)

