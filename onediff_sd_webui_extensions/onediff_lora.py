import torch
import oneflow as flow
from onediff.infer_compiler.deployable_module import DeployableModule


class HijackLoraActivate:
    def __init__(self, conv_dict=None):
        from modules import extra_networks
        self.conv_dict = conv_dict

        if "lora" in extra_networks.extra_network_registry:
            cls_extra_network_lora = type(extra_networks.extra_network_registry["lora"])
        else:
            cls_extra_network_lora = None
        self.lora_class = cls_extra_network_lora

    def __enter__(self):
        if self.lora_class is None:
            return
        self.orig_func = self.lora_class.activate
        self.lora_class.activate = hijacked_activate(self.lora_class.activate, conv_dict=self.conv_dict)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lora_class is None:
            return
        self.lora_class.activate = self.orig_func
        self.lora_class = None
        self.orig_func = None


def hijacked_activate(activate_func, *, conv_dict=None):
    import networks

    if hasattr(activate_func, "_onediff_hijacked"):
        return activate_func

    def activate(self, p, params_list):
        activate_func(self, p, params_list)
        if isinstance(p.sd_model.model.diffusion_model, DeployableModule):
            onediff_sd_model: DeployableModule = p.sd_model.model.diffusion_model
            for name, sub_module in onediff_sd_model.named_modules():
                if not isinstance(
                    sub_module,
                    (
                        torch.nn.Linear,
                        torch.nn.Conv2d,
                        torch.nn.GroupNorm,
                        torch.nn.LayerNorm,
                    ),
                ):
                    continue
                networks.network_apply_weights(sub_module)

                # for LyCORIS cases
                if conv_dict is not None and isinstance(sub_module, torch.nn.Conv2d):
                    target_tensor = conv_dict.get(name + ".weight", None)
                    if target_tensor is None:
                        continue
                    target_tensor.copy_(
                        flow.utils.tensor.from_torch(sub_module.weight.permute(0, 2, 3, 1))
                    )

    activate._onediff_hijacked = True
    return activate
