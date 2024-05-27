import torch


def should_patch_torch_module(torch_module):
    # https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/ops.py#L132-L134
    supported_types = (
        torch.nn.ConvTranspose2d,
        torch.nn.GroupNorm,
        torch.nn.Conv3d,
        torch.nn.Conv2d,
        torch.nn.Linear,
    )
    full_cls_name = (
        type(torch_module).__module__ + "." + type(torch_module).__qualname__
    )
    return (
        isinstance(torch_module, supported_types)
        or full_cls_name.endswith("controlnet.ControlLoraOps.Linear")
        or full_cls_name.endswith("controlnet.ControlLoraOps.Conv2d")
    )


def apply_comfy_settings(torch_module, flow_module):
    keys = [
        "comfy_cast_inputs",
        "comfy_cast_weights",
        "weight_function",
        "bias_function",
    ]
    for k in keys:
        setattr(flow_module, k, getattr(torch_module, k, None))


class PatchForComfy:
    def __init__(self, oneflow_model):
        self.oneflow_model = oneflow_model

    def patch(
        self,
        torch_module,
        should_patch: callable = None,
        apply_settings: callable = None,
    ):
        if should_patch(torch_module):
            apply_settings(torch_module, self.oneflow_model)

    def __call__(self, torch_module):
        self.patch(torch_module, should_patch_torch_module, apply_comfy_settings)
