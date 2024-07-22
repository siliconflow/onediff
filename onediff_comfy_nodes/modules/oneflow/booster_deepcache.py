from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Optional

from comfy.model_patcher import ModelPatcher

from ..booster_interface import BoosterExecutor
from .utils.booster_utils import set_compiled_options
from .utils.deep_cache_speedup import deep_cache_speedup
from .utils.graph_path import generate_graph_path


@dataclass
class DeepcacheBoosterExecutor(BoosterExecutor):
    cache_interval: int = 3
    cache_layer_id: int = 0
    cache_block_id: int = 1
    start_step: int = 0
    end_step: int = 1000

    @singledispatchmethod
    def execute(self, model):
        print(
            "Warning: DeepcacheBoosterExecutor.apply is not implemented for model type:",
            type(model),
        )
        return model

    @execute.register(ModelPatcher)
    def _(
        self, model, ckpt_name: Optional[str] = None, use_graph=True, **kwargs
    ) -> ModelPatcher:
        model = deep_cache_speedup(
            model=model,
            use_graph=use_graph,
            cache_interval=self.cache_interval,
            cache_layer_id=self.cache_layer_id,
            cache_block_id=self.cache_block_id,
            start_step=self.start_step,
            end_step=self.end_step,
            use_oneflow_deepcache_speedup_modelpatcher=False,
        )[0]
        if ckpt_name:
            graph_file = generate_graph_path(
                ckpt_name, model.fast_deep_cache_unet._torch_module
            )
            set_compiled_options(model.fast_deep_cache_unet, graph_file)
            graph_file = generate_graph_path(
                ckpt_name, model.deep_cache_unet._torch_module
            )
            set_compiled_options(model.deep_cache_unet, graph_file)

        return model
