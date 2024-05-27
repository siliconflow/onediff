from packaging import version
from typing import Union, Dict
from collections import defaultdict

import torch
from onediff.infer_compiler import DeployableModule
from onediff.utils import logger
from diffusers.models.lora import (
    LoRACompatibleConv,
    LoRACompatibleLinear,
)
from .utils import fuse_lora, get_adapter_names, is_peft_available
from diffusers.utils import is_accelerate_available

if is_peft_available():
    import peft

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module
USE_PEFT_BACKEND = False


def load_lora_into_unet(
    self,
    state_dict,
    network_alphas,
    unet,
    low_cpu_mem_usage=None,
    adapter_name=None,
    _pipeline=None,
    *,
    lora_scale: float = 1.0,
    offload_device="cpu",
    use_cache=False,
):
    if adapter_name is None:
        adapter_name = get_adapter_names(unet)

    if hasattr(unet, "adapter_names"):
        if adapter_name in unet.adapter_names:
            raise ValueError(
                f"[OneDiffX load_lora_into_unet] The adapter name {adapter_name} already exists in UNet"
            )
        else:
            unet.adapter_name.add(adapter_name)
            unet.active_adapter_name[adapter_name] = 1.0
    else:
        unet.adapter_name = set([adapter_name])
        unet.active_adapter_name = {adapter_name: 1.0}

    keys = list(state_dict.keys())
    cls = type(self)

    if all(
        key.startswith(cls.unet_name) or key.startswith(cls.text_encoder_name)
        for key in keys
    ):
        # Load the layers corresponding to UNet.
        logger.info(f"Loading {cls.unet_name}.")

        unet_keys = [k for k in keys if k.startswith(cls.unet_name)]
        state_dict = {
            k.replace(f"{cls.unet_name}.", ""): v
            for k, v in state_dict.items()
            if k in unet_keys
        }

        if network_alphas is not None:
            alpha_keys = [
                k for k in network_alphas.keys() if k.startswith(cls.unet_name)
            ]
            network_alphas = {
                k.replace(f"{cls.unet_name}.", ""): v
                for k, v in network_alphas.items()
                if k in alpha_keys
            }

    else:
        # Otherwise, we're dealing with the old format. This means the `state_dict` should only
        # contain the module names of the `unet` as its keys WITHOUT any prefix.
        warn_message = "You have saved the LoRA weights using the old format. To convert the old LoRA weights to the new format, you can first load them in a dictionary and then create a new dictionary like the following: `new_state_dict = {f'unet.{module_name}': params for module_name, params in old_state_dict.items()}`."
        logger.warning(warn_message)

    # unet.load_attn
    _load_attn_procs(
        unet,
        state_dict,
        network_alphas=network_alphas,
        low_cpu_mem_usage=low_cpu_mem_usage,
        adapter_name=adapter_name,
        _pipeline=_pipeline,
        lora_scale=lora_scale,
        offload_device=offload_device,
        use_cache=use_cache,
    )


# The code is referenced from https://github.com/huggingface/diffusers/blob/ce9825b56bd8a6849e68b9590022e935400659e6/src/diffusers/loaders/unet.py#L383
def _convert_state_dict_legacy_attn_format(self, state_dict, network_alphas):
    is_new_lora_format = all(
        key.startswith(self.unet_name) or key.startswith(self.text_encoder_name)
        for key in state_dict.keys()
    )
    if is_new_lora_format:
        # Strip the `"unet"` prefix.
        is_text_encoder_present = any(
            key.startswith(self.text_encoder_name) for key in state_dict.keys()
        )
        if is_text_encoder_present:
            warn_message = "The state_dict contains LoRA params corresponding to the text encoder which are not being used here. To use both UNet and text encoder related LoRA params, use [`pipe.load_lora_weights()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.load_lora_weights)."
            logger.warning(warn_message)
        unet_keys = [k for k in state_dict.keys() if k.startswith(self.unet_name)]
        state_dict = {
            k.replace(f"{self.unet_name}.", ""): v
            for k, v in state_dict.items()
            if k in unet_keys
        }

    # change processor format to 'pure' LoRACompatibleLinear format
    if any("processor" in k.split(".") for k in state_dict.keys()):

        def format_to_lora_compatible(key):
            if "processor" not in key.split("."):
                return key
            return (
                key.replace(".processor", "")
                .replace("to_out_lora", "to_out.0.lora")
                .replace("_lora", ".lora")
            )

        state_dict = {format_to_lora_compatible(k): v for k, v in state_dict.items()}

        if network_alphas is not None:
            network_alphas = {
                format_to_lora_compatible(k): v for k, v in network_alphas.items()
            }
    return state_dict, network_alphas


def _load_attn_procs(
    self,
    pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
    **kwargs,
):

    lora_scale = kwargs.pop("lora_scale", 1.0)
    offload_device = kwargs.pop("offload_device", "cpu")
    use_cache = kwargs.pop("use_cache", False)
    _pipeline = kwargs.pop("_pipeline", None)
    network_alphas = kwargs.pop("network_alphas", None)
    adapter_name = kwargs.pop("adapter_name", None)
    state_dict = pretrained_model_name_or_path_or_dict

    is_network_alphas_none = network_alphas is None

    is_lora = (
        all(("lora" in k or k.endswith(".alpha")) for k in state_dict.keys())
        and not USE_PEFT_BACKEND
    )

    if is_lora:
        # correct keys
        if hasattr(self, "convert_state_dict_legacy_attn_format"):
            state_dict, network_alphas = self.convert_state_dict_legacy_attn_format(
                state_dict, network_alphas
            )
        else:
            state_dict, network_alphas = _convert_state_dict_legacy_attn_format(
                self, state_dict, network_alphas
            )

        if network_alphas is not None:
            network_alphas_keys = list(network_alphas.keys())
            used_network_alphas_keys = set()

        lora_grouped_dict = defaultdict(dict)
        mapped_network_alphas = {}

        all_keys = list(state_dict.keys())
        for key in all_keys:
            value = state_dict.pop(key)
            attn_processor_key, sub_key = (
                ".".join(key.split(".")[:-3]),
                ".".join(key.split(".")[-3:]),
            )
            lora_grouped_dict[attn_processor_key][sub_key] = value

            # Create another `mapped_network_alphas` dictionary so that we can properly map them.
            if network_alphas is not None:
                for k in network_alphas_keys:
                    if k.replace(".alpha", "") in key:
                        mapped_network_alphas.update(
                            {attn_processor_key: network_alphas.get(k)}
                        )
                        used_network_alphas_keys.add(k)

        if not is_network_alphas_none:
            if len(set(network_alphas_keys) - used_network_alphas_keys) > 0:
                raise ValueError(
                    f"[OneDiffX _load_attn_procs] The `network_alphas` has to be empty at this point but has the following keys \n\n {', '.join(network_alphas.keys())}"
                )

        if len(state_dict) > 0:
            raise ValueError(
                f"[OneDiffX _load_attn_procs] The `state_dict` has to be empty at this point but has the following keys \n\n {', '.join(state_dict.keys())}"
            )

        for key, value_dict in lora_grouped_dict.items():
            if isinstance(self, DeployableModule):
                attn_processor = self._torch_module
            else:
                attn_processor = self
            for sub_key in key.split("."):
                attn_processor = getattr(attn_processor, sub_key)

            # Process non-attention layers, which don't have to_{k,v,q,out_proj}_lora layers
            # or add_{k,v,q,out_proj}_proj_lora layers.
            rank = value_dict["lora.down.weight"].shape[0]

            if isinstance(
                attn_processor,
                (
                    LoRACompatibleConv,
                    torch.nn.Conv2d,
                    LoRACompatibleLinear,
                    torch.nn.Linear,
                ),
            ):
                fuse_lora(
                    attn_processor,
                    value_dict,
                    lora_scale,
                    mapped_network_alphas.get(key),
                    rank,
                    offload_device=offload_device,
                    adapter_name=adapter_name,
                )
            elif is_peft_available() and isinstance(
                attn_processor,
                (peft.tuners.lora.layer.Linear, peft.tuners.lora.layer.Conv2d),
            ):
                fuse_lora(
                    attn_processor.base_layer,
                    value_dict,
                    lora_scale,
                    mapped_network_alphas.get(key),
                    rank,
                    offload_device=offload_device,
                    adapter_name=adapter_name,
                )
            else:
                raise ValueError(
                    f"[OneDiffX _load_attn_procs] Module {key} is not a Conv2d or Linear module, got type {type(attn_processor)}"
                )
    else:
        raise ValueError(
            f"[OneDiffX _load_attn_procs] {pretrained_model_name_or_path_or_dict} does not seem to be in the correct format expected by LoRA training."
        )

    is_model_cpu_offload = False
    is_sequential_cpu_offload = False

    if not USE_PEFT_BACKEND:
        if _pipeline is not None:
            for _, component in _pipeline.components.items():
                if isinstance(component, torch.nn.Module) and hasattr(
                    component, "_hf_hook"
                ):
                    is_model_cpu_offload = isinstance(
                        getattr(component, "_hf_hook"), CpuOffload
                    )
                    is_sequential_cpu_offload = isinstance(
                        getattr(component, "_hf_hook"), AlignDevicesHook
                    )

                    logger.info(
                        "Accelerate hooks detected. Since you have called `load_lora_weights()`, the previous hooks will be first removed. Then the LoRA parameters will be loaded and the hooks will be applied again."
                    )
                    remove_hook_from_module(
                        component, recurse=is_sequential_cpu_offload
                    )

        # self.to(dtype=self.dtype, device=self.device)

        # Offload back.
        if is_model_cpu_offload:
            _pipeline.enable_model_cpu_offload()
        elif is_sequential_cpu_offload:
            _pipeline.enable_sequential_cpu_offload()
        # Unsafe code />
