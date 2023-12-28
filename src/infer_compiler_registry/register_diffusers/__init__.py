from onediff.infer_compiler.transform import register

# register(package_names=["diffusers"])
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.models.transformer_2d import Transformer2DModel
from diffusers.models.resnet import SpatioTemporalResBlock
from diffusers.models.attention import TemporalBasicTransformerBlock
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import TemporalDecoder

from .attention_processor_oflow import Attention as AttentionOflow
from .attention_processor_oflow import AttnProcessor as AttnProcessorOflow
from .attention_processor_oflow import LoRAAttnProcessor2_0 as LoRAAttnProcessorOflow
from .transformer_2d_oflow import Transformer2DModel as Transformer2DModelOflow
from .transformer_3d_oflow import (
    SpatioTemporalResBlock as SpatioTemporalResBlockOflow,
)
from .transformer_3d_oflow import (
    TemporalBasicTransformerBlock as TemporalBasicTransformerBlockOflow,
)
from .transformer_3d_oflow import TemporalDecoder as TemporalDecoderOflow

torch2oflow_class_map = {
    Attention: AttentionOflow,
    AttnProcessor2_0: AttnProcessorOflow,
    LoRAAttnProcessor2_0: LoRAAttnProcessorOflow,
    SpatioTemporalResBlock: SpatioTemporalResBlockOflow,
    TemporalBasicTransformerBlock: TemporalBasicTransformerBlockOflow,
    TemporalDecoder: TemporalDecoderOflow,
}
if Transformer2DModelOflow is not None:
    torch2oflow_class_map.update({Transformer2DModel: Transformer2DModelOflow})

register(torch2oflow_class_map=torch2oflow_class_map)
