from collections import defaultdict
from packaging import version

import torch
import diffusers

if version.parse(diffusers.__version__) >= version.parse("0.22.0"):
    from diffusers.utils import convert_state_dict_to_diffusers
else:
    from .state_dict_utils import convert_state_dict_to_diffusers

if version.parse(diffusers.__version__) >= version.parse("0.24.0"):
    from diffusers.models.lora import (
        text_encoder_attn_modules,
        text_encoder_mlp_modules,
    )
else:
    from diffusers.loaders import text_encoder_attn_modules, text_encoder_mlp_modules
from diffusers.utils import is_accelerate_available

from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT
from onediff.infer_compiler.utils.log_utils import logger

from .utils import fuse_lora, get_adapter_names

USE_PEFT_BACKEND = False

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module

# The code is mainly referenced from https://github.com/huggingface/diffusers/blob/b09b90e24c7ef0252a1a587939972c2e02d305a6/src/diffusers/loaders/lora.py#L485
def load_lora_into_text_encoder(
    cls,
    state_dict,
    network_alphas,
    text_encoder,
    prefix=None,
    lora_scale=1.0,
    low_cpu_mem_usage=None,
    adapter_name=None,
    _pipeline=None,
):
    """
    This will load and fuse the LoRA layers specified in `state_dict` into `text_encoder`

    Parameters:
        cls (`LoraLoaderMixin`):
            The pipeline to load lora into its text encoder
        state_dict (`dict`):
            A standard state dict containing the lora layer parameters. The key should be prefixed with an
            additional `text_encoder` to distinguish between unet lora layers.
        network_alphas (`Dict[str, float]`):
            See `LoRALinearLayer` for more details.
        text_encoder (`CLIPTextModel`):
            The text encoder model to load the LoRA layers into.
        prefix (`str`):
            Expected prefix of the `text_encoder` in the `state_dict`.
        lora_scale (`float`):
            How much to scale the output of the lora linear layer before it is added with the output of the regular
            lora layer.
        low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
            Speed up model loading only loading the pretrained weights and not initializing the weights. This also
            tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
            Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
            argument to `True` will raise an error.
        adapter_name (`str`, *optional*):
            Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
            `default_{i}` where i is the total number of adapters being loaded.
    """
    low_cpu_mem_usage = (
        low_cpu_mem_usage if low_cpu_mem_usage is not None else _LOW_CPU_MEM_USAGE_DEFAULT
    )

    if adapter_name is None:
        adapter_name = get_adapter_names(text_encoder)

    if hasattr(text_encoder, "adapter_names"):
        if adapter_name in text_encoder.adapter_names:
            raise ValueError(
                f"[OneDiffX load_lora_into_text_encoder] The adapter name {adapter_name} already exists in text_encoder"
            )
        else:
            text_encoder.adapter_name.add(adapter_name)
            text_encoder.active_adapter_name[adapter_name] = 1.0
    else:
        text_encoder.adapter_name = set([adapter_name])
        text_encoder.active_adapter_name = {adapter_name: 1.0}

    # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
    # then the `state_dict` keys should have `self.unet_name` and/or `self.text_encoder_name` as
    # their prefixes.
    keys = list(state_dict.keys())
    prefix = cls.text_encoder_name if prefix is None else prefix

    # Safe prefix to check with.
    if any(cls.text_encoder_name in key for key in keys):
        # Load the layers corresponding to text encoder and make necessary adjustments.
        text_encoder_keys = [k for k in keys if k.startswith(prefix) and k.split(".")[0] == prefix]
        text_encoder_lora_state_dict = {
            k.replace(f"{prefix}.", ""): v for k, v in state_dict.items() if k in text_encoder_keys
        }

        if len(text_encoder_lora_state_dict) > 0:
            logger.info(f"Loading {prefix}.")
            rank = {}
            text_encoder_lora_state_dict = convert_state_dict_to_diffusers(
                text_encoder_lora_state_dict
            )

            if USE_PEFT_BACKEND:
                # convert state dict
                text_encoder_lora_state_dict = convert_state_dict_to_peft(
                    text_encoder_lora_state_dict
                )

                for name, _ in text_encoder_attn_modules(text_encoder):
                    rank_key = f"{name}.out_proj.lora_B.weight"
                    rank[rank_key] = text_encoder_lora_state_dict[rank_key].shape[1]

                patch_mlp = any(".mlp." in key for key in text_encoder_lora_state_dict.keys())
                if patch_mlp:
                    for name, _ in text_encoder_mlp_modules(text_encoder):
                        rank_key_fc1 = f"{name}.fc1.lora_B.weight"
                        rank_key_fc2 = f"{name}.fc2.lora_B.weight"

                        rank[rank_key_fc1] = text_encoder_lora_state_dict[rank_key_fc1].shape[1]
                        rank[rank_key_fc2] = text_encoder_lora_state_dict[rank_key_fc2].shape[1]
            else:
                for name, _ in text_encoder_attn_modules(text_encoder):
                    rank_key = f"{name}.out_proj.lora_linear_layer.up.weight"
                    rank.update({rank_key: text_encoder_lora_state_dict[rank_key].shape[1]})

                patch_mlp = any(".mlp." in key for key in text_encoder_lora_state_dict.keys())
                if patch_mlp:
                    for name, _ in text_encoder_mlp_modules(text_encoder):
                        rank_key_fc1 = f"{name}.fc1.lora_linear_layer.up.weight"
                        rank_key_fc2 = f"{name}.fc2.lora_linear_layer.up.weight"
                        rank[rank_key_fc1] = text_encoder_lora_state_dict[rank_key_fc1].shape[1]
                        rank[rank_key_fc2] = text_encoder_lora_state_dict[rank_key_fc2].shape[1]

            # group text encoder lora state_dict
            te_lora_grouped_dict = defaultdict(dict)
            for k, v in text_encoder_lora_state_dict.items():
                name_elems = k.split(".")
                module_name, lora_module_name = (
                    ".".join(name_elems[:6]),
                    ".".join(name_elems[6:]),
                )
                te_lora_grouped_dict[module_name][lora_module_name] = v

            if network_alphas is not None:
                alpha_keys = [
                    k
                    for k in network_alphas.keys()
                    if k.startswith(prefix) and k.split(".")[0] == prefix
                ]
                network_alphas = {
                    k.replace(f"{prefix}.", ""): v
                    for k, v in network_alphas.items()
                    if k in alpha_keys
                }

            if USE_PEFT_BACKEND:
                from peft import LoraConfig

                lora_config_kwargs = get_peft_kwargs(
                    rank, network_alphas, text_encoder_lora_state_dict, is_unet=False
                )

                lora_config = LoraConfig(**lora_config_kwargs)

                # adapter_name
                if adapter_name is None:
                    adapter_name = get_adapter_name(text_encoder)

                (
                    is_model_cpu_offload,
                    is_sequential_cpu_offload,
                ) = cls._optionally_disable_offloading(_pipeline)

                # inject LoRA layers and load the state dict
                # in transformers we automatically check whether the adapter name is already in use or not
                text_encoder.load_adapter(
                    adapter_name=adapter_name,
                    adapter_state_dict=text_encoder_lora_state_dict,
                    peft_config=lora_config,
                )

                # scale LoRA layers with `lora_scale`
                scale_lora_layers(text_encoder, weight=lora_scale)
            else:

                network_alphas = {} if network_alphas is None else network_alphas
                is_network_alphas_populated = len(network_alphas) > 0

                for name, attn_module in text_encoder_attn_modules(text_encoder):
                    query_alpha = network_alphas.pop(name + ".to_q_lora.down.weight.alpha", None)
                    key_alpha = network_alphas.pop(name + ".to_k_lora.down.weight.alpha", None)
                    value_alpha = network_alphas.pop(name + ".to_v_lora.down.weight.alpha", None)
                    out_alpha = network_alphas.pop(name + ".to_out_lora.down.weight.alpha", None)

                    if isinstance(rank, dict):
                        current_rank = rank.pop(f"{name}.out_proj.lora_linear_layer.up.weight")
                    else:
                        current_rank = rank

                    fuse_lora(
                        attn_module.q_proj,
                        te_lora_grouped_dict.pop(f"{name}.q_proj"),
                        lora_scale,
                        query_alpha,
                        current_rank,
                        adapter_name=adapter_name,
                        prefix="lora_linear_layer",
                    )
                    fuse_lora(
                        attn_module.k_proj,
                        te_lora_grouped_dict.pop(f"{name}.k_proj"),
                        lora_scale,
                        key_alpha,
                        current_rank,
                        adapter_name=adapter_name,
                        prefix="lora_linear_layer",
                    )
                    fuse_lora(
                        attn_module.v_proj,
                        te_lora_grouped_dict.pop(f"{name}.v_proj"),
                        lora_scale,
                        value_alpha,
                        current_rank,
                        adapter_name=adapter_name,
                        prefix="lora_linear_layer",
                    )
                    fuse_lora(
                        attn_module.out_proj,
                        te_lora_grouped_dict.pop(f"{name}.out_proj"),
                        lora_scale,
                        out_alpha,
                        current_rank,
                        adapter_name=adapter_name,
                        prefix="lora_linear_layer",
                    )

                if patch_mlp:
                    for name, mlp_module in text_encoder_mlp_modules(text_encoder):
                        fc1_alpha = network_alphas.pop(
                            name + ".fc1.lora_linear_layer.down.weight.alpha", None
                        )
                        fc2_alpha = network_alphas.pop(
                            name + ".fc2.lora_linear_layer.down.weight.alpha", None
                        )

                        current_rank_fc1 = rank.pop(f"{name}.fc1.lora_linear_layer.up.weight")
                        current_rank_fc2 = rank.pop(f"{name}.fc2.lora_linear_layer.up.weight")

                        fuse_lora(
                            mlp_module.fc1,
                            te_lora_grouped_dict.pop(f"{name}.fc1"),
                            lora_scale,
                            fc1_alpha,
                            current_rank_fc1,
                            adapter_name=adapter_name,
                            prefix="lora_linear_layer",
                        )
                        fuse_lora(
                            mlp_module.fc2,
                            te_lora_grouped_dict.pop(f"{name}.fc2"),
                            lora_scale,
                            fc2_alpha,
                            current_rank_fc2,
                            adapter_name=adapter_name,
                            prefix="lora_linear_layer",
                        )

                if is_network_alphas_populated and len(network_alphas) > 0:
                    raise ValueError(
                        f"The `network_alphas` has to be empty at this point but has the following keys \n\n {', '.join(network_alphas.keys())}"
                    )

                is_pipeline_offloaded = _pipeline is not None and any(
                    isinstance(c, torch.nn.Module) and hasattr(c, "_hf_hook")
                    for c in _pipeline.components.values()
                )
                if is_pipeline_offloaded and low_cpu_mem_usage:
                    low_cpu_mem_usage = True
                    logger.info(
                        f"Pipeline {_pipeline.__class__} is offloaded. Therefore low cpu mem usage loading is forced."
                    )
                unexpected_keys = list(te_lora_grouped_dict.keys())
                if len(unexpected_keys) != 0:
                    raise ValueError(
                        f"failed to load text encoder state dict, unexpected keys: {unexpected_keys}"
                    )

                # <Unsafe code
                # We can be sure that the following works as all we do is change the dtype and device of the text encoder
                # Now we remove any existing hooks to
                is_model_cpu_offload = False
                is_sequential_cpu_offload = False
                if _pipeline is not None:
                    for _, component in _pipeline.components.items():
                        if isinstance(component, torch.nn.Module):
                            if hasattr(component, "_hf_hook"):
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

            text_encoder.to(device=text_encoder.device, dtype=text_encoder.dtype)

            # Offload back.
            if is_model_cpu_offload:
                _pipeline.enable_model_cpu_offload()
            elif is_sequential_cpu_offload:
                _pipeline.enable_sequential_cpu_offload()
            # Unsafe code />
