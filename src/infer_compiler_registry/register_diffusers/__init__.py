from onediff.infer_compiler.transform import register

from packaging import version
import importlib.metadata

diffusers_version = version.parse(importlib.metadata.version("diffusers"))

# register(package_names=["diffusers"])
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.models.transformer_2d import Transformer2DModel
from diffusers.models.resnet import SpatioTemporalResBlock
if diffusers_version >= version.parse("0.25.00"):
    from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import TemporalDecoder
else:
    from diffusers.models.autoencoder_kl_temporal_decoder import TemporalDecoder

from .attention_processor_oflow import Attention as AttentionOflow
from .attention_processor_oflow import AttnProcessor as AttnProcessorOflow
from .attention_processor_oflow import LoRAAttnProcessor2_0 as LoRAAttnProcessorOflow
from .transformer_2d_oflow import Transformer2DModel as Transformer2DModelOflow
from .spatio_temporal_oflow import (
    SpatioTemporalResBlock as SpatioTemporalResBlockOflow,
)
from .spatio_temporal_oflow import TemporalDecoder as TemporalDecoderOflow

torch2oflow_class_map = {
    Attention: AttentionOflow,
    AttnProcessor2_0: AttnProcessorOflow,
    LoRAAttnProcessor2_0: LoRAAttnProcessorOflow,
    SpatioTemporalResBlock: SpatioTemporalResBlockOflow,
    TemporalDecoder: TemporalDecoderOflow,
}
if Transformer2DModelOflow is not None:
    torch2oflow_class_map.update({Transformer2DModel: Transformer2DModelOflow})

register(torch2oflow_class_map=torch2oflow_class_map)
