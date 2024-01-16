from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple, List
from collections import OrderedDict

import torch
import safetensors.torch

# 1. cache (limit 10 LoRAs)
# 2. load lora and add to cache
# 3. unet load cache and cache offload

def load_state_dict_cached(pretrained_model_name_or_path_or_dict) -> Tuple[Dict, Any]:
    from_cached = False
    if not isinstance(pretrained_model_name_or_path_or_dict, (str, Path)):
        return pretrained_model_name_or_path_or_dict, from_cached

    if not isinstance(pretrained_model_name_or_path_or_dict, Path):
        pretrained_model_name_or_path_or_dict = Path(pretrained_model_name_or_path_or_dict)
    if not Path(pretrained_model_name_or_path_or_dict).exists():
        return pretrained_model_name_or_path_or_dict, from_cached

    from_cached = True
    global cached_loras
    if pretrained_model_name_or_path_or_dict in cached_loras:
        return cached_loras[pretrained_model_name_or_path_or_dict], from_cached
    
    state_dict = safetensors.torch.load_file(pretrained_model_name_or_path_or_dict, device=0)
    cached_loras[pretrained_model_name_or_path_or_dict] = state_dict
    return state_dict, from_cached



def load_lora_weights(
        pipeline,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[str] = None,
        **kwargs,
    ):
    pretrained_model_name_or_path_or_dict, _ = load_state_dict_cached(pretrained_model_name_or_path_or_dict)
    pipeline.load_lora_weights(pretrained_model_name_or_path_or_dict, adapter_name, **kwargs)




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