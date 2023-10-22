# ONEDIFF_MODEL_CLASS_REPLACEMENT_MAP = { PYTORCH_MODEL_CLASS: ONEFLOW_MODEL_CLASS }
# ONEDIFF_CUSTOM_TORCH2OF_FUNC_TYPE_MAP = { Function :  TYPE }
import comfy

from .attention import CrossAttentionPytorch as CrossAttentionPytorch1f
from .attention import SpatialTransformer as SpatialTransformer1f
from .linear import Linear as Linear1f


ONEDIFF_MODEL_CLASS_REPLACEMENT_MAP = {
    comfy.ldm.modules.attention.CrossAttentionPytorch: CrossAttentionPytorch1f,
    comfy.ldm.modules.attention.SpatialTransformer: SpatialTransformer1f,
    comfy.ops.Linear: Linear1f,
}

#    import comfy
#         from .mock_comfy import CrossAttentionPytorch, SpatialTransformer, Linear

#         cls_key = get_mock_cls_name(comfy.ldm.modules.attention.CrossAttentionPytorch)
#         update_class_proxies({cls_key: CrossAttentionPytorch})

#         cls_key = get_mock_cls_name(comfy.ops.Linear)
#         update_class_proxies({cls_key: Linear})

#         cls_key = get_mock_cls_name(comfy.ldm.modules.attention.SpatialTransformer)
#         update_class_proxies({cls_key: SpatialTransformer})

#         @torch2of.register
#         def _(mod: comfy.latent_formats.SDXL, verbose=False):
#             return default_converter(mod, verbose=verbose)
