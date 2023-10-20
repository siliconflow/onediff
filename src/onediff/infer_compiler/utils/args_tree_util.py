import transformers

from ..convert_torch_to_of.proxy import proxy_class

_ONEFLOW_HAS_REGISTER_RELAXED_TYPE_API = False
try:
    from oneflow.framework.args_tree import register_relaxed_type

    _ONEFLOW_HAS_REGISTER_RELAXED_TYPE_API = True
except:
    pass


def register_args_tree_relaxed_types():
    if _ONEFLOW_HAS_REGISTER_RELAXED_TYPE_API:
        register_relaxed_type(
            proxy_class(transformers.modeling_outputs.BaseModelOutputWithPooling)
        )
        register_relaxed_type(
            proxy_class(transformers.models.clip.modeling_clip.CLIPTextModelOutput)
        )
    else:
        pass
