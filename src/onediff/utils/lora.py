from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
from collections import OrderedDict

import torch
import safetensors.torch

from onediff.infer_compiler.utils.log_utils import logger


def load_state_dict_cached(lora_name) -> Dict:
    if not isinstance(lora_name, (str, Path)):
        return lora_name

    if not isinstance(lora_name, Path):
        lora_name = Path(lora_name)
    if not Path(lora_name).exists():
        return lora_name

    global cached_loras
    if lora_name in cached_loras:
        logger.debug(f"[OneDiff Cached LoRA] get cached lora of name: {str(lora_name)}")
        return cached_loras[lora_name]

    state_dict = safetensors.torch.load_file(lora_name)
    cached_loras[lora_name] = state_dict
    logger.debug(f"[OneDiff Cached LoRA] create cached lora of name: {str(lora_name)}")
    return state_dict


def load_lora_weights(
    pipeline,
    pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
    adapter_name: Optional[str] = None,
    **kwargs,
):
    pretrained_model_name_or_path_or_dict = load_state_dict_cached(
        pretrained_model_name_or_path_or_dict
    )
    pipeline.load_lora_weights(
        pretrained_model_name_or_path_or_dict, adapter_name, **kwargs
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


cached_loras = LRUCacheDict(10)
