from onediff.infer_compiler.backends.oneflow import online_quantization_utils
from register_comfy.CrossAttentionPatch import is_crossAttention_patch

from .patch_executor import PatchExecutorBase


def convert_to_nested(original_dict):
    new_dict = {}

    for key, value in original_dict.items():
        if isinstance(value, dict):
            new_value = convert_to_nested(value)
        elif is_crossAttention_patch(value):
            new_value = value.get_bind_model()
        else:
            new_value = value

        # Update the new dictionary with the processed value
        new_dict[key] = new_value
    return new_dict


class QuantizedInputPatch(PatchExecutorBase):
    def __init__(self):
        self.is_use_patch = False

    def set_patch(self):
        if self.check_patch():
            return

        def new_patch_input_adapter(in_args, in_kwargs):
            return in_args, convert_to_nested(in_kwargs)

        self.is_use_patch = True
        online_quantization_utils.patch_input_adapter = new_patch_input_adapter

    def get_patch(self, module):
        pass

    def check_patch(self):
        return self.is_use_patch
