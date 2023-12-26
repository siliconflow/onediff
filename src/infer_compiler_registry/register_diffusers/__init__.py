from onediff.infer_compiler.transform import register

# register(package_names=["diffusers"])
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.models.transformer_2d import Transformer2DModel
from diffusers.models.transformer_temporal import TransformerSpatioTemporalModel
from diffusers.models.resnet import SpatioTemporalResBlock
from diffusers.models.attention import TemporalBasicTransformerBlock
from diffusers.models.autoencoder_kl_temporal_decoder import TemporalDecoder

from .attention_processor_oflow import Attention as AttentionOflow
from .attention_processor_oflow import AttnProcessor as AttnProcessorOflow
from .attention_processor_oflow import LoRAAttnProcessor2_0 as LoRAAttnProcessorOflow
from .transformer_2d_oflow import Transformer2DModel as Transformer2DModelOflow
from .transformer_spatial_oflow import (
    TransformerSpatioTemporalModel as TransformerSpatioTemporalModelOflow,
)
from .transformer_spatial_oflow import (
    SpatioTemporalResBlock as SpatioTemporalResBlockOflow,
)
from .transformer_spatial_oflow import (
    TemporalBasicTransformerBlock as TemporalBasicTransformerBlockOflow,
)
from .transformer_spatial_oflow import TemporalDecoder as TemporalDecoderOflow

torch2oflow_class_map = {
    Attention: AttentionOflow,
    AttnProcessor2_0: AttnProcessorOflow,
    LoRAAttnProcessor2_0: LoRAAttnProcessorOflow,
    TransformerSpatioTemporalModel: TransformerSpatioTemporalModelOflow,
    SpatioTemporalResBlock: SpatioTemporalResBlockOflow,
    TemporalBasicTransformerBlock: TemporalBasicTransformerBlockOflow,
    TemporalDecoder: TemporalDecoderOflow,
}
if Transformer2DModelOflow is not None:
    torch2oflow_class_map.update({Transformer2DModel: Transformer2DModelOflow})

register(torch2oflow_class_map=torch2oflow_class_map)


from onediff.infer_compiler.transform import transform_mgr
from onediff.infer_compiler.transform.builtin_transform import proxy_class


_ONEFLOW_HAS_REGISTER_RELAXED_TYPE_API = False
try:
    from oneflow.framework.args_tree import register_relaxed_type

    _ONEFLOW_HAS_REGISTER_RELAXED_TYPE_API = True
except ImportError:
    pass


def register_args_tree_relaxed_types():
    transformers_mocked = False
    for pkg_name in transform_mgr.get_mocked_packages():
        if "transformers" in pkg_name:
            transformers_mocked = True
            break

    if _ONEFLOW_HAS_REGISTER_RELAXED_TYPE_API and transformers_mocked:
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        from transformers.models.clip.modeling_clip import CLIPTextModelOutput

        register_relaxed_type(proxy_class(BaseModelOutputWithPooling))
        register_relaxed_type(proxy_class(CLIPTextModelOutput))
    else:
        pass


register_args_tree_relaxed_types()
