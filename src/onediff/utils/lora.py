from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
from collections import OrderedDict, defaultdict
from contextlib import nullcontext

import torch

from onediff.infer_compiler.utils.log_utils import logger
from .profiler import with_cProfile

from diffusers.loaders.lora import LoraLoaderMixin
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.utils import DIFFUSERS_CACHE, HF_HUB_OFFLINE
from diffusers.models.modeling_utils import (
    _LOW_CPU_MEM_USAGE_DEFAULT,
    load_model_dict_into_meta,
)
from diffusers.utils import is_accelerate_available

if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module


USE_PEFT_BACKEND = False


def linear_fuse_lora(
    self: torch.nn.Linear,
    state_dict: Dict[str, torch.Tensor],
    lora_scale: float = 1.0,
    alpha: float = None,
    rank: float = None,
):
    assert isinstance(self, torch.nn.Linear)
    linear_unfuse_lora(self)
    dtype, device = self.weight.data.dtype, self.weight.data.device

    self._lora_up = state_dict["lora.up.weight"]
    self._lora_down = state_dict["lora.down.weight"]
    self._lora_scale = lora_scale

    w_down = state_dict["lora.down.weight"].float().to(device)
    w_up = state_dict["lora.up.weight"].float().to(device)

    if alpha is not None:
        w_up = w_up * alpha / rank

    lora_weight = lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0]
    fused_weight = self.weight.data.float() + lora_weight
    self.weight.data.copy_(fused_weight.to(device=device, dtype=dtype))
    print(w_up.shape)


def linear_unfuse_lora(self: torch.nn.Linear):
    assert isinstance(self, torch.nn.Linear)
    if not hasattr(self, "_lora_up") or self._lora_up is None:
        return

    fused_weight = self.weight.data
    dtype, device = fused_weight.dtype, fused_weight.device

    w_up = self._lora_up.to(device=device).float()
    w_down = self._lora_down.to(device).float()

    unfused_weight = self.weight.data.float() - (
        self._lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0]
    )
    self.weight.data.copy_(unfused_weight.to(device=device, dtype=dtype))

    self._lora_up = None
    self._lora_down = None
    self._lora_scale = None


def conv_fuse_lora(
    self: torch.nn.Conv2d,
    state_dict: Dict[str, torch.Tensor],
    lora_scale: float = 1.0,
    alpha: float = None,
    rank: float = None,
) -> None:
    assert isinstance(self, torch.nn.Conv2d)
    conv_unfuse_lora(self)
    dtype, device = self.weight.data.dtype, self.weight.data.device

    self._lora_up = state_dict["lora.up.weight"]
    self._lora_down = state_dict["lora.down.weight"]
    self._lora_scale = lora_scale

    w_down = state_dict["lora.down.weight"].float().to(device)
    w_up = state_dict["lora.up.weight"].float().to(device)

    if alpha is not None:
        w_up = w_up * alpha / rank

    lora_weight = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
    lora_weight = lora_weight.reshape((self.weight.shape)) * lora_scale

    fused_weight = self.weight.data.float() + lora_weight
    self.weight.data.copy_(fused_weight.to(device=device, dtype=dtype))


def conv_unfuse_lora(self: torch.nn.Conv2d):
    assert isinstance(self, torch.nn.Conv2d)
    if not hasattr(self, "_lora_up") or self._lora_up is None:
        return

    fused_weight = self.weight.data
    dtype, device = fused_weight.data.dtype, fused_weight.data.device

    w_up = self._lora_up.to(device=device).float()
    w_down = self._lora_down.to(device).float()

    fusion = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
    fusion = fusion.reshape((fused_weight.shape))
    unfused_weight = fused_weight.float() - (self._lora_scale * fusion)
    self.weight.data.copy_(unfused_weight.to(device=device, dtype=dtype))

    self._lora_up = None
    self._lora_down = None
    self._lora_scale = None


# @with_cProfile()
def load_and_fuse_lora(
    self: LoraLoaderMixin,
    lora: Union[str, Path, Dict[str, torch.Tensor]],
    lora_scale: float = 1.0,
    adapter_name: Optional[str] = None,
    **kwargs,
) -> None:
    state_dict, network_alphas = load_state_dict_cached(lora, unet_config=self.unet.config, **kwargs)

    is_correct_format = all("lora" in key for key in state_dict.keys())
    if not is_correct_format:
        raise ValueError("Invalid LoRA checkpoint.")

    # load lora into unet
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
            logger.warn(warn_message)

    # unet.load_attn

    low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
    # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
    # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
    _pipeline = kwargs.pop("_pipeline", None)

    is_network_alphas_none = network_alphas is None

    is_lora = (
        all(("lora" in k or k.endswith(".alpha")) for k in state_dict.keys())
        and not USE_PEFT_BACKEND
    )
    if is_lora:
        # correct keys
        state_dict, network_alphas = self.unet.convert_state_dict_legacy_attn_format(
            state_dict, network_alphas
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
                    f"The `network_alphas` has to be empty at this point but has the following keys \n\n {', '.join(network_alphas.keys())}"
                )

        if len(state_dict) > 0:
            raise ValueError(
                f"The `state_dict` has to be empty at this point but has the following keys \n\n {', '.join(state_dict.keys())}"
            )

        for key, value_dict in lora_grouped_dict.items():
            attn_processor = self.unet
            for sub_key in key.split("."):
                attn_processor = getattr(attn_processor, sub_key)

            # Process non-attention layers, which don't have to_{k,v,q,out_proj}_lora layers
            # or add_{k,v,q,out_proj}_proj_lora layers.
            rank = value_dict["lora.down.weight"].shape[0]

            if isinstance(attn_processor, LoRACompatibleConv):
                ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
                with ctx():
                    conv_fuse_lora(
                        attn_processor,
                        value_dict,
                        lora_scale,
                        mapped_network_alphas.get(key),
                        rank,
                    )
            elif isinstance(attn_processor, LoRACompatibleLinear):
                ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
                with ctx():
                    linear_fuse_lora(
                        attn_processor,
                        value_dict,
                        lora_scale,
                        mapped_network_alphas.get(key),
                        rank,
                    )
            else:
                raise ValueError(
                    f"Module {key} is not a LoRACompatibleConv or LoRACompatibleLinear module."
                )
    else:
        raise
        # value_dict = {k.replace("lora.", ""): v for k, v in value_dict.items()}
        # lora_layers_list.append((attn_processor, lora))

        # if low_cpu_mem_usage:
        #     device = next(iter(value_dict.values())).device
        #     dtype = next(iter(value_dict.values())).dtype
        #     load_model_dict_into_meta(lora, value_dict, device=device, dtype=dtype)
        # else:
        #     lora.load_state_dict(value_dict)

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

        # only custom diffusion needs to set attn processors
        # if is_custom_diffusion:
        #     self.set_attn_processor(attn_processors)

        # set lora layers
        # for target_module, lora_layer in lora_layers_list:
        #     target_module.set_lora_layer(lora_layer)

        self.to(dtype=self.dtype, device=self.device)

        # Offload back.
        if is_model_cpu_offload:
            _pipeline.enable_model_cpu_offload()
        elif is_sequential_cpu_offload:
            _pipeline.enable_sequential_cpu_offload()
        # Unsafe code />

    # load lora weights
    text_encoder_state_dict = {
        k: v for k, v in state_dict.items() if "text_encoder." in k
    }
    if len(text_encoder_state_dict) > 0:
        self.load_lora_into_text_encoder(
            text_encoder_state_dict,
            network_alphas=network_alphas,
            text_encoder=self.text_encoder,
            prefix="text_encoder",
            lora_scale=self.lora_scale,
            adapter_name=adapter_name,
            _pipeline=self,
        )

    text_encoder_2_state_dict = {
        k: v for k, v in state_dict.items() if "text_encoder_2." in k
    }
    if len(text_encoder_2_state_dict) > 0:
        self.load_lora_into_text_encoder(
            text_encoder_2_state_dict,
            network_alphas=network_alphas,
            text_encoder=self.text_encoder_2,
            prefix="text_encoder_2",
            lora_scale=self.lora_scale,
            adapter_name=adapter_name,
            _pipeline=self,
        )


class LRUCacheDict(OrderedDict):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if len(self) >= self.capacity:
            oldest_key = next(iter(self))
            del self[oldest_key]
        super().__setitem__(key, value)


def load_state_dict_cached(
    lora: Union[str, Path, Dict[str, torch.Tensor]], **kwargs,
) -> Tuple[Dict, Dict]:
    assert isinstance(lora, (str, Path, dict))
    if isinstance(lora, dict):
        state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(lora, **kwargs)
        return state_dict, network_alphas

    global CachedLoRAs
    weight_name = kwargs.get("weight_name", None)

    lora_name = str(lora) + (f"/{weight_name}" if weight_name else "")
    if lora_name in CachedLoRAs:
        logger.debug(f"[OneDiff Cached LoRA] get cached lora of name: {str(lora_name)}")
        return CachedLoRAs[lora_name]

    state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(lora, **kwargs,)
    CachedLoRAs[lora_name] = (state_dict, network_alphas)
    logger.debug(f"[OneDiff Cached LoRA] create cached lora of name: {str(lora_name)}")
    return state_dict, network_alphas


CachedLoRAs = LRUCacheDict(100)
