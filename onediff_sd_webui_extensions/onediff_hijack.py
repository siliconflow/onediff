from typing import Any, Mapping

import torch
from modules import sd_models
from modules.sd_hijack_utils import CondFunc
from onediff_shared import onediff_enabled


def unload_model_weights(sd_model=None, info=None):
    from modules import devices, lowvram, shared

    m = sd_model or shared.sd_model
    if m.lowvram:
        lowvram.send_everything_to_cpu()
    else:
        m.to(devices.cpu)
    devices.torch_gc()
    return sd_model


def send_model_to_cpu(m):
    # do nothing
    pass


def hijack_function(module, name, new_name, new_value):
    # restore original function in case of reload
    unhijack_function(module=module, name=name, new_name=new_name)
    setattr(module, new_name, getattr(module, name))
    setattr(module, name, new_value)


def unhijack_function(module, name, new_name):
    if hasattr(module, new_name):
        setattr(module, name, getattr(module, new_name))
        delattr(module, new_name)


def do_hijack():
    from modules import script_callbacks, sd_models

    script_callbacks.on_script_unloaded(undo_hijack)
    hijack_function(
        module=sd_models,
        name="unload_model_weights",
        new_name="__onediff_original_unload_model_weights",
        new_value=unload_model_weights,
    )
    hijack_function(
        module=sd_models,
        name="send_model_to_cpu",
        new_name="__onediff_original_send_model_to_cpu",
        new_value=send_model_to_cpu,
    )


def undo_hijack():
    from modules import sd_models

    unhijack_function(
        module=sd_models,
        name="unload_model_weights",
        new_name="__onediff_original_unload_model_weights",
    )
    unhijack_function(
        module=sd_models,
        name="send_model_to_cpu",
        new_name="__onediff_original_send_model_to_cpu",
    )


def onediff_hijack_load_model_weights(
    orig_func, model, checkpoint_info: sd_models.CheckpointInfo, state_dict: dict, timer
):
    # load_model_weights(model, checkpoint_info: CheckpointInfo, state_dict, timer)
    sd_model_hash = checkpoint_info.calculate_shorthash()
    import onediff_shared

    if onediff_shared.current_unet_graph.sha == sd_model_hash:
        model.model.diffusion_model = onediff_shared.current_unet_graph.graph_module
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("model.diffusion_model.")
        }

        # for stable-diffusion-webui/modules/sd_models.py:load_model_weights model.is_ssd check
        state_dict[
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight"
        ] = model.get_parameter(
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight"
        )
    return orig_func(model, checkpoint_info, state_dict, timer)


def onediff_hijack_load_state_dict(
    orig_func,
    self,
    state_dict: Mapping[str, Any],
    strict: bool = True,
    assign: bool = False,
):
    if (
        len(state_dict) > 0
        and next(iter(state_dict.values())).is_cuda
        and next(self.parameters()).is_meta
    ):
        return orig_func(self, state_dict, strict, assign=True)
    else:
        return orig_func(self, state_dict, strict, assign)


# fmt: off
def onediff_hijaced_LoadStateDictOnMeta___enter__(orig_func, self):
    from modules import shared
    if shared.cmd_opts.disable_model_loading_ram_optimization:
        return

    sd = self.state_dict
    device = self.device

    def load_from_state_dict(original, module, state_dict, prefix, *args, **kwargs):
        used_param_keys = []

        for name, param in module._parameters.items():
            if param is None:
                continue

            key = prefix + name
            sd_param = sd.pop(key, None)
            if sd_param is not None:
                state_dict[key] = sd_param.to(dtype=self.get_weight_dtype(key))
                used_param_keys.append(key)

            if param.is_meta:
                dtype = sd_param.dtype if sd_param is not None else param.dtype
                module._parameters[name] = torch.nn.parameter.Parameter(torch.zeros_like(param, device=device, dtype=dtype), requires_grad=param.requires_grad)

        for name in module._buffers:
            key = prefix + name

            sd_param = sd.pop(key, None)
            if sd_param is not None:
                state_dict[key] = sd_param
                used_param_keys.append(key)

        original(module, state_dict, prefix, *args, **kwargs)

        for key in used_param_keys:
            state_dict.pop(key, None)

    # def load_state_dict(original, module, state_dict, strict=True):
    def load_state_dict(original, module, state_dict, strict=True):
        """torch makes a lot of copies of the dictionary with weights, so just deleting entries from state_dict does not help
        because the same values are stored in multiple copies of the dict. The trick used here is to give torch a dict with
        all weights on meta device, i.e. deleted, and then it doesn't matter how many copies torch makes.

        In _load_from_state_dict, the correct weight will be obtained from a single dict with the right weights (sd).

        The dangerous thing about this is if _load_from_state_dict is not called, (if some exotic module overloads
        the function and does not call the original) the state dict will just fail to load because weights
        would be on the meta device.
        """

        if state_dict is sd:
            state_dict = {k: v.to(device="meta", dtype=v.dtype) for k, v in state_dict.items()}

        # ------------------- DIFF HERE -------------------
        # original(module, state_dict, strict=strict)
        if len(state_dict) > 0 and next(iter(state_dict.values())).is_cuda and next(module.parameters()).is_meta:
            assign = True
        else:
            assign = False
        # orig_func(original, module, state_dict, strict=strict, assign=assign)
        original(module, state_dict, strict=strict, assign=assign)

    module_load_state_dict = self.replace(torch.nn.Module, 'load_state_dict', lambda *args, **kwargs: load_state_dict(module_load_state_dict, *args, **kwargs))
    module_load_from_state_dict = self.replace(torch.nn.Module, '_load_from_state_dict', lambda *args, **kwargs: load_from_state_dict(module_load_from_state_dict, *args, **kwargs))
    linear_load_from_state_dict = self.replace(torch.nn.Linear, '_load_from_state_dict', lambda *args, **kwargs: load_from_state_dict(linear_load_from_state_dict, *args, **kwargs))
    conv2d_load_from_state_dict = self.replace(torch.nn.Conv2d, '_load_from_state_dict', lambda *args, **kwargs: load_from_state_dict(conv2d_load_from_state_dict, *args, **kwargs))
    mha_load_from_state_dict = self.replace(torch.nn.MultiheadAttention, '_load_from_state_dict', lambda *args, **kwargs: load_from_state_dict(mha_load_from_state_dict, *args, **kwargs))
    layer_norm_load_from_state_dict = self.replace(torch.nn.LayerNorm, '_load_from_state_dict', lambda *args, **kwargs: load_from_state_dict(layer_norm_load_from_state_dict, *args, **kwargs))
    group_norm_load_from_state_dict = self.replace(torch.nn.GroupNorm, '_load_from_state_dict', lambda *args, **kwargs: load_from_state_dict(group_norm_load_from_state_dict, *args, **kwargs))
# fmt: on


CondFunc(
    "modules.sd_disable_initialization.LoadStateDictOnMeta.__enter__",
    onediff_hijaced_LoadStateDictOnMeta___enter__,
    lambda _, *args, **kwargs: onediff_enabled,
)
CondFunc(
    "modules.sd_models.load_model_weights",
    onediff_hijack_load_model_weights,
    lambda _, *args, **kwargs: onediff_enabled,
)
