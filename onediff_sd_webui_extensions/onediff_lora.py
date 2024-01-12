import torch
from onediff.infer_compiler.with_oneflow_compile import DeployableModule

def hijacked_activate(activate_func):
    import networks
    def activate(self, p, params_list):
        activate_func(self, p, params_list)
        if isinstance(p.sd_model.model.diffusion_model, DeployableModule):
            self.switch_from_onediff = True
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
    return activate
