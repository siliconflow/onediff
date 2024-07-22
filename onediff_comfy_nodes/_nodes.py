import uuid
from typing import Optional, Tuple

import folder_paths
import torch
from nodes import CheckpointLoaderSimple, ControlNetLoader
from onediff.utils.import_utils import (  # type: ignore
    is_nexfort_available,
    is_oneflow_available,
)

from ._config import is_disable_oneflow_backend
from .modules import BoosterExecutor, BoosterScheduler, BoosterSettings

if is_oneflow_available() and not is_disable_oneflow_backend():
    from .modules.oneflow import BasicOneFlowBoosterExecutor

    BasicBoosterExecutor = BasicOneFlowBoosterExecutor
    print("\033[1;31mUsing OneFlow backend\033[0m (Default)")
elif is_nexfort_available():
    from .modules.nexfort.booster_basic import BasicNexFortBoosterExecutor

    BasicBoosterExecutor = BasicNexFortBoosterExecutor
    print("\033[1;32mUsing Nexfort backend\033[0m (Default)")
else:
    raise RuntimeError(
        "Neither OneFlow nor Nexfort is available. Please ensure at least one of them is installed."
    )

__all__ = [
    "ModelSpeedup",
    "VaeSpeedup",
    "ControlnetSpeedup",
    "OneDiffApplyModelBooster",
    "OneDiffControlNetLoader",
    "OneDiffCheckpointLoaderSimple",
]


class SpeedupMixin:
    """A mix-in class to provide speedup functionality."""

    FUNCTION = "speedup"
    CATEGORY = "OneDiff"

    @torch.inference_mode()
    def speedup(
        self,
        model,
        inplace: bool = False,
        custom_booster: Optional[BoosterScheduler] = None,
        booster_settings: Optional[BoosterSettings] = None,
        *args,
        **kwargs
    ) -> Tuple:
        """
        Speed up the model inference.

        Args:
            model: The input model to be sped up.
            inplace (bool, optional): Whether to perform the operation inplace. Defaults to False.
            custom_booster (BoosterScheduler, optional): Custom booster scheduler to use. Defaults to None.
            *args: Additional positional arguments to be passed to the underlying functions.
            **kwargs: Additional keyword arguments to be passed to the underlying functions.

        Returns:
            Tuple: Tuple containing the optimized model.
        """
        if booster_settings is None and not hasattr(self, "booster_settings"):
            self.booster_settings = BoosterSettings(tmp_cache_key=str(uuid.uuid4()))

        if custom_booster:
            booster = custom_booster
            booster.inplace = inplace
        else:
            booster = BoosterScheduler(BasicBoosterExecutor(), inplace=inplace)
        booster.settings = (
            self.booster_settings if booster_settings is None else booster_settings
        )
        return (booster(model, *args, **kwargs),)


class ModelSpeedup(SpeedupMixin):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "custom_booster": ("CUSTOM_BOOSTER",),
                "inplace": (
                    "BOOLEAN",
                    {"default": True, "label_on": "yes", "label_off": "no"},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)


class VaeSpeedup(SpeedupMixin):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"vae": ("VAE",)},
            "optional": {
                "custom_booster": ("CUSTOM_BOOSTER",),
                "inplace": (
                    "BOOLEAN",
                    {"default": True, "label_on": "yes", "label_off": "no"},
                ),
            },
        }

    RETURN_TYPES = ("VAE",)

    def speedup(self, vae, inplace=False, custom_booster: BoosterScheduler = None):
        return super().speedup(vae, inplace, custom_booster)


class ControlnetSpeedup(SpeedupMixin):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
            },
            "optional": {
                "inplace": (
                    "BOOLEAN",
                    {"default": True, "label_on": "yes", "label_off": "no"},
                ),
                "custom_booster": ("CUSTOM_BOOSTER",),
            },
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "speedup"
    CATEGORY = "OneDiff"

    def speedup(
        self,
        control_net=None,
        inplace=True,
        custom_booster: BoosterScheduler = None,
        **kwargs
    ):
        return super().speedup(control_net, inplace, custom_booster)


class OneDiffApplyModelBooster:
    """Main class responsible for optimizing models."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "quantization_booster": ("QuantizationBooster",),
                "deepcache_booster": ("DeepCacheBooster",),
                "torchcompile_booster": ("TorchCompileBooster",),
            },
        }

    CATEGORY = "OneDiff/Booster"
    RETURN_TYPES = ("CUSTOM_BOOSTER",)
    FUNCTION = "speedup_module"

    @torch.no_grad()
    def speedup_module(
        self,
        quantization_booster: BoosterExecutor = None,
        deepcache_booster=None,
        torchcompile_booster=None,
    ):
        """Apply the optimization technique to the model."""
        booster_executors = []
        if deepcache_booster:
            booster_executors.append(deepcache_booster)
        if quantization_booster:
            booster_executors.append(quantization_booster)
        if torchcompile_booster:
            booster_executors.append(torchcompile_booster)

        assert len(booster_executors) > 0
        return (BoosterScheduler(booster_executors),)


class OneDiffControlNetLoader(ControlNetLoader):
    @classmethod
    def INPUT_TYPES(s):
        ret = super().INPUT_TYPES()
        ret.update(
            {
                "optional": {
                    "custom_booster": ("CUSTOM_BOOSTER",),
                }
            }
        )
        return ret

    CATEGORY = "OneDiff/Loaders"
    FUNCTION = "onediff_load_controlnet"

    @torch.inference_mode()
    def onediff_load_controlnet(self, control_net_name, custom_booster=None):
        controlnet = super().load_controlnet(control_net_name)[0]
        if custom_booster is None:
            custom_booster = BoosterScheduler(BasicBoosterExecutor())
        controlnet = custom_booster(controlnet, ckpt_name=control_net_name)

        return (controlnet,)


class OneDiffCheckpointLoaderSimple(CheckpointLoaderSimple, SpeedupMixin):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_speedup": (["disable", "enable"],),
            },
            "optional": {
                "custom_booster": ("CUSTOM_BOOSTER",),
            },
        }

    CATEGORY = "OneDiff/Loaders"
    FUNCTION = "onediff_load_checkpoint"

    def __init__(self) -> None:
        super().__init__()
        self.unet_booster_settings = BoosterSettings(tmp_cache_key=str(uuid.uuid4()))
        self.vae_booster_settings = BoosterSettings(tmp_cache_key=str(uuid.uuid4()))

    @torch.inference_mode()
    def onediff_load_checkpoint(
        self,
        ckpt_name,
        vae_speedup="disable",
        custom_booster: BoosterScheduler = None,
    ):
        modelpatcher, clip, vae = self.load_checkpoint(ckpt_name)
        modelpatcher = self.speedup(
            modelpatcher,
            inplace=True,
            custom_booster=custom_booster,
            booster_settings=self.unet_booster_settings,
        )[0]

        if vae_speedup == "enable":
            vae = self.speedup(
                vae,
                inplace=True,
                custom_booster=custom_booster,
                booster_settings=self.vae_booster_settings,
            )[0]

        # Set weight inplace update
        modelpatcher.weight_inplace_update = True
        return (
            modelpatcher,
            clip,
            vae,
        )
