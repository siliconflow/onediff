from modules import extra_networks, shared
import networks
import logging
from onediff.infer_compiler.with_oneflow_compile import DeployableModule
from typing import Union
import torch
from modules import shared, devices, sd_models, errors, scripts, sd_hijack


class ExtraNetworkLora(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('lora')

        self.errors = {}
        """mapping of network names to the number of errors the network had during operation"""

    def activate(self, p, params_list):
        additional = shared.opts.sd_lora

        self.errors.clear()

        if additional != "None" and additional in networks.available_networks and not any(x for x in params_list if x.items[0] == additional):
            p.all_prompts = [x + f"<lora:{additional}:{shared.opts.extra_networks_default_multiplier}>" for x in p.all_prompts]
            params_list.append(extra_networks.ExtraNetworkParams(items=[additional, shared.opts.extra_networks_default_multiplier]))

        names = []
        te_multipliers = []
        unet_multipliers = []
        dyn_dims = []
        for params in params_list:
            assert params.items

            names.append(params.positional[0])

            te_multiplier = float(params.positional[1]) if len(params.positional) > 1 else 1.0
            te_multiplier = float(params.named.get("te", te_multiplier))

            unet_multiplier = float(params.positional[2]) if len(params.positional) > 2 else te_multiplier
            unet_multiplier = float(params.named.get("unet", unet_multiplier))

            dyn_dim = int(params.positional[3]) if len(params.positional) > 3 else None
            dyn_dim = int(params.named["dyn"]) if "dyn" in params.named else dyn_dim

            te_multipliers.append(te_multiplier)
            unet_multipliers.append(unet_multiplier)
            dyn_dims.append(dyn_dim)

        networks.load_networks(names, te_multipliers, unet_multipliers, dyn_dims)

        if shared.opts.lora_add_hashes_to_infotext:
            network_hashes = []
            for item in networks.loaded_networks:
                shorthash = item.network_on_disk.shorthash
                if not shorthash:
                    continue

                alias = item.mentioned_name
                if not alias:
                    continue

                alias = alias.replace(":", "").replace(",", "")

                network_hashes.append(f"{alias}: {shorthash}")

            if network_hashes:
                p.extra_generation_params["Lora hashes"] = ", ".join(network_hashes)

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
                onediff_network_apply_weights(sub_module)

    def deactivate(self, p):
        if self.errors:
            p.comment("Networks with errors: " + ", ".join(f"{k} ({v})" for k, v in self.errors.items()))

            self.errors.clear()

        if isinstance(shared.sd_model.model.diffusion_model, DeployableModule):
            onediff_sd_model: DeployableModule = p.sd_model.model.diffusion_model
            for name, sub_module in onediff_sd_model.named_modules():
                if not isinstance(
                    sub_module,
                    (
                        torch.nn.Linear,
                        torch.nn.Conv2d,
                        torch.nn.GroupNorm,
                        torch.nn.LayerNorm,
                        torch.nn.MultiheadAttention,
                    ),
                ):
                    continue
                with torch.no_grad():
                    onediff_network_restore_weights_from_backup(sub_module)


def onediff_network_apply_weights(
    self: Union[
        torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm
    ]
):
    """
    Applies the currently selected set of networks to the weights of torch layer self.
    If weights already have this particular set of networks applied, does nothing.
    If not, restores orginal weights from backup and alters weights according to networks.
    """

    network_layer_name = getattr(self, "network_layer_name", None)
    if network_layer_name is None:
        return

    current_names = getattr(self, "network_current_names", ())
    wanted_names = tuple(
        (x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim)
        for x in networks.loaded_networks
    )

    weights_backup = getattr(self, "network_weights_backup", None)
    if weights_backup is None and wanted_names != ():
        if current_names != ():
            raise RuntimeError(
                "no backup weights found and current weights are not unchanged"
            )

        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (self.in_proj_weight.to(devices.cpu, copy=True), self.out_proj.weight.to(devices.cpu, copy=True))
        else:
            weights_backup = self.weight.to(devices.cpu, copy=True)

        self.network_weights_backup = weights_backup

    bias_backup = getattr(self, "network_bias_backup", None)
    if bias_backup is None:
        if isinstance(self, torch.nn.MultiheadAttention) and self.out_proj.bias is not None:
            bias_backup = self.out_proj.bias.to(devices.cpu, copy=True)
        elif getattr(self, 'bias', None) is not None:
            bias_backup = self.bias.to(devices.cpu, copy=True)
        else:
            bias_backup = None
        self.network_bias_backup = bias_backup

    if current_names != wanted_names:
        onediff_network_restore_weights_from_backup(self)

        for net in networks.loaded_networks:
            module = net.modules.get(network_layer_name, None)
            if module is not None and hasattr(self, "weight"):
                try:
                    with torch.no_grad():
                        updown, ex_bias = module.calc_updown(self.weight)

                        if len(self.weight.shape) == 4 and self.weight.shape[1] == 9:
                            # inpainting model. zero pad updown to make channel[1]  4 to 9
                            updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5))

                        self.weight += updown
                        if ex_bias is not None and hasattr(self, "bias"):
                            if self.bias is None:
                                self.bias = torch.nn.Parameter(ex_bias)
                            else:
                                self.bias += ex_bias
                except RuntimeError as e:
                    logging.debug(f"Network {net.name} layer {network_layer_name}: {e}")
                    networks.extra_network_lora.errors[net.name] = (
                        networks.extra_network_lora.errors.get(net.name, 0) + 1
                    )

                continue

            module_q = net.modules.get(network_layer_name + "_q_proj", None)
            module_k = net.modules.get(network_layer_name + "_k_proj", None)
            module_v = net.modules.get(network_layer_name + "_v_proj", None)
            module_out = net.modules.get(network_layer_name + "_out_proj", None)

            if isinstance(self, torch.nn.MultiheadAttention) and module_q and module_k and module_v and module_out:
                try:
                    with torch.no_grad():
                        updown_q, _ = module_q.calc_updown(self.in_proj_weight)
                        updown_k, _ = module_k.calc_updown(self.in_proj_weight)
                        updown_v, _ = module_v.calc_updown(self.in_proj_weight)
                        updown_qkv = torch.vstack([updown_q, updown_k, updown_v])
                        updown_out, ex_bias = module_out.calc_updown(
                            self.out_proj.weight
                        )

                        self.in_proj_weight += updown_qkv
                        self.out_proj.weight += updown_out
                    if ex_bias is not None:
                        if self.out_proj.bias is None:
                            self.out_proj.bias = torch.nn.Parameter(ex_bias)
                        else:
                            self.out_proj.bias += ex_bias

                except RuntimeError as e:
                    logging.debug(f"Network {net.name} layer {network_layer_name}: {e}")
                    networks.extra_network_lora.errors[net.name] = (
                        networks.extra_network_lora.errors.get(net.name, 0) + 1
                    )

                continue

            if module is None:
                continue

            logging.debug(
                f"Network {net.name} layer {network_layer_name}: couldn't find supported operation"
            )
            networks.extra_network_lora.errors[net.name] = (
                networks.extra_network_lora.errors.get(net.name, 0) + 1
            )

        self.network_current_names = wanted_names

def onediff_network_restore_weights_from_backup(
    self: Union[
        torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention
    ]
):
    weights_backup = getattr(self, "network_weights_backup", None)
    bias_backup = getattr(self, "network_bias_backup", None)

    if weights_backup is None and bias_backup is None:
        return

    if weights_backup is not None:
        if isinstance(self, torch.nn.MultiheadAttention):
            self.in_proj_weight.copy_(weights_backup[0])
            self.out_proj.weight.copy_(weights_backup[1])
        else:
            self.weight.copy_(weights_backup)

    if bias_backup is not None:
        if isinstance(self, torch.nn.MultiheadAttention):
            self.out_proj.bias.copy_(bias_backup)
        else:
            self.bias.copy_(bias_backup)
    else:
        if isinstance(self, torch.nn.MultiheadAttention):
            self.out_proj.bias = None
        else:
            self.bias = None
