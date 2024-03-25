from onediff.infer_compiler.transform import register

from packaging import version
import importlib.metadata

diffusers_version = version.parse(importlib.metadata.version("diffusers"))

# register(package_names=["diffusers"])
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
)
from diffusers.models.attention_processor import LoRAAttnProcessor2_0

if diffusers_version < version.parse("0.26.00"):
    from diffusers.models.unet_2d_condition import UNet2DConditionModel
    from diffusers.models.unet_2d_blocks import AttnUpBlock2D, CrossAttnUpBlock2D, UpBlock2D
    from diffusers.models.transformer_2d import Transformer2DModel
else:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
    from diffusers.models.unets.unet_2d_blocks import AttnUpBlock2D, CrossAttnUpBlock2D, UpBlock2D
    from diffusers.models.transformers.transformer_2d import Transformer2DModel

if diffusers_version >= version.parse("0.25.00"):
    from diffusers.models.upsampling import Upsample2D
else:
    from diffusers.models.resnet import Upsample2D
if diffusers_version >= version.parse("0.24.00"):
    from diffusers.models.resnet import SpatioTemporalResBlock
    from diffusers.models.attention import TemporalBasicTransformerBlock

    if diffusers_version >= version.parse("0.26.00"):
        from diffusers.models.unets.unet_spatio_temporal_condition import (
            UNetSpatioTemporalConditionModel,
        )
        from diffusers.models.transformers.transformer_temporal import TransformerSpatioTemporalModel
    else:
        from diffusers.models.transformer_temporal import TransformerSpatioTemporalModel
        from diffusers.models.unet_spatio_temporal_condition import (
            UNetSpatioTemporalConditionModel,
        )

    if diffusers_version >= version.parse("0.25.00"):
        from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import (
            TemporalDecoder,
        )
    else:
        from diffusers.models.autoencoder_kl_temporal_decoder import TemporalDecoder
    
    from .spatio_temporal_oflow import SpatioTemporalResBlock as SpatioTemporalResBlockOflow
    from .spatio_temporal_oflow import TemporalDecoder as TemporalDecoderOflow
    from .spatio_temporal_oflow import (
        TransformerSpatioTemporalModel as TransformerSpatioTemporalModelOflow,
    )
    from .spatio_temporal_oflow import (
        TemporalBasicTransformerBlock as TemporalBasicTransformerBlockOflow,
    )
    from .spatio_temporal_oflow import (
        UNetSpatioTemporalConditionModel as UNetSpatioTemporalConditionModelOflow,
    )

from .attention_processor_oflow import Attention as AttentionOflow
from .attention_processor_oflow import AttnProcessor as AttnProcessorOflow
from .attention_processor_oflow import LoRAAttnProcessor2_0 as LoRAAttnProcessorOflow
from .unet_2d_condition_oflow import UNet2DConditionModel as UNet2DConditionModelOflow
from .unet_2d_blocks_oflow import AttnUpBlock2D as AttnUpBlock2DOflow
from .unet_2d_blocks_oflow import CrossAttnUpBlock2D as CrossAttnUpBlock2DOflow
from .unet_2d_blocks_oflow import UpBlock2D as UpBlock2DOflow
from .resnet_oflow import Upsample2D as Upsample2DOflow
from .transformer_2d_oflow import Transformer2DModel as Transformer2DModelOflow

# For CI
if diffusers_version >= version.parse("0.24.00"):
    torch2oflow_class_map = {
        Attention: AttentionOflow,
        AttnProcessor: AttnProcessorOflow,
        AttnProcessor2_0: AttnProcessorOflow,
        LoRAAttnProcessor2_0: LoRAAttnProcessorOflow,
        SpatioTemporalResBlock: SpatioTemporalResBlockOflow,
        TemporalDecoder: TemporalDecoderOflow,
        TransformerSpatioTemporalModel: TransformerSpatioTemporalModelOflow,
        TemporalBasicTransformerBlock: TemporalBasicTransformerBlockOflow,
        UNetSpatioTemporalConditionModel: UNetSpatioTemporalConditionModelOflow,
    }
else:
    torch2oflow_class_map = {
        Attention: AttentionOflow,
        AttnProcessor: AttnProcessorOflow,
        AttnProcessor2_0: AttnProcessorOflow,
        LoRAAttnProcessor2_0: LoRAAttnProcessorOflow,
    }

torch2oflow_class_map.update({Transformer2DModel: Transformer2DModelOflow})
torch2oflow_class_map.update({UNet2DConditionModel: UNet2DConditionModelOflow})
torch2oflow_class_map.update({AttnUpBlock2D: AttnUpBlock2DOflow})
torch2oflow_class_map.update({CrossAttnUpBlock2D: CrossAttnUpBlock2DOflow})
torch2oflow_class_map.update({UpBlock2D: UpBlock2DOflow})
torch2oflow_class_map.update({Upsample2D: Upsample2DOflow})

register(torch2oflow_class_map=torch2oflow_class_map)
