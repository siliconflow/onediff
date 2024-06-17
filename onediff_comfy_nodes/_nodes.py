from typing import Optional, Tuple
import folder_paths
import torch
import comfy
import uuid
from nodes import CheckpointLoaderSimple, ControlNetLoader
from ._config import is_disable_oneflow_backend
from .modules import BoosterScheduler, BoosterExecutor, BoosterSettings
from onediff.utils.import_utils import is_nexfort_available  # type: ignore
from onediff.utils.import_utils import is_oneflow_available

if is_oneflow_available() and not is_disable_oneflow_backend():
    from .modules.oneflow import BasicOneFlowBoosterExecutor

    BasicBoosterExecutor = BasicOneFlowBoosterExecutor
elif is_nexfort_available():
    from .modules.nexfort.booster_basic import BasicNexFortBoosterExecutor

    BasicBoosterExecutor = BasicNexFortBoosterExecutor
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
        if not hasattr(self, "booster_settings"):
            self.booster_settings = BoosterSettings(tmp_cache_key=str(uuid.uuid4()))

        if custom_booster:
            booster = custom_booster
            booster.inplace = inplace
        else:
            booster = BoosterScheduler(BasicBoosterExecutor(), inplace=inplace)
        booster.settings = self.booster_settings
        return (booster(model, *args, **kwargs),)


class ModelSpeedup(SpeedupMixin):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"model": ("MODEL",), "inplace": ([False, True],),},
            "optional": {"custom_booster": ("CUSTOM_BOOSTER",),},
        }

    RETURN_TYPES = ("MODEL",)


class VaeSpeedup(SpeedupMixin):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"vae": ("VAE",), "inplace": ([False, True],),},
            "optional": {"custom_booster": ("CUSTOM_BOOSTER",),},
        }

    RETURN_TYPES = ("VAE",)

    def speedup(self, vae, inplace=False, custom_booster: BoosterScheduler = None):
        return super().speedup(vae, inplace, custom_booster)


class ControlnetSpeedup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "control_net": ("CONTROL_NET",),
                "cnet_stack": ("CONTROL_NET_STACK",),
                "custom_booster": ("CUSTOM_BOOSTER",),
            },
        }

    RETURN_TYPES = (
        "CONTROL_NET",
        "CONTROL_NET_STACK",
    )
    FUNCTION = "speedup"
    CATEGORY = "OneDiff"

    @torch.no_grad()
    def speedup(
        self, control_net=None, cnet_stack=[], custom_booster: BoosterScheduler = None
    ):
        if custom_booster:
            booster = custom_booster
        else:
            booster = BoosterScheduler(BasicBoosterExecutor(), inplace=True)

        if control_net:
            control_net = booster(control_net)

        new_cnet_stack = []
        for cnet in cnet_stack:
            new_cnet = tuple([booster(cnet[0])] + list(cnet[1:]))
            new_cnet_stack.append(new_cnet)
        return (
            control_net,
            new_cnet_stack,
        )


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
        ret.update({"optional": {"custom_booster": ("CUSTOM_BOOSTER",),}})
        return ret

    CATEGORY = "OneDiff/Loaders"
    FUNCTION = "onediff_load_controlnet"

    @torch.no_grad()
    def onediff_load_controlnet(self, control_net_name, custom_booster=None):
        controlnet = super().load_controlnet(control_net_name)[0]
        if custom_booster is None:
            custom_booster = BoosterScheduler(BasicBoosterExecutor())
        controlnet = custom_booster(controlnet, ckpt_name=control_net_name)

        return (controlnet,)


class OneDiffCheckpointLoaderSimple(CheckpointLoaderSimple):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_speedup": (["disable", "enable"],),
            },
            "optional": {"custom_booster": ("CUSTOM_BOOSTER",),},
        }

    CATEGORY = "OneDiff/Loaders"
    FUNCTION = "onediff_load_checkpoint"

    @staticmethod
    def _load_checkpoint(
        ckpt_name, vae_speedup="disable", custom_booster: BoosterScheduler = None
    ):
        """Loads a checkpoint, applying speedup techniques."""

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        # Unpack outputs
        modelpatcher, clip, vae = out[:3]

        # Apply custom booster if provided, otherwise use a basic one
        custom_booster = custom_booster or BoosterScheduler(BasicBoosterExecutor())
        modelpatcher = custom_booster(modelpatcher, ckpt_name=ckpt_name)

        # Apply VAE speedup if enabled
        if vae_speedup == "enable":
            vae = BoosterScheduler(BasicBoosterExecutor())(vae, ckpt_name=ckpt_name)

        # Set weight inplace update
        modelpatcher.weight_inplace_update = True

        return modelpatcher, clip, vae

    @torch.inference_mode()
    def onediff_load_checkpoint(
        self, ckpt_name, vae_speedup="disable", custom_booster: BoosterScheduler = None,
    ):
        out = self._load_checkpoint(ckpt_name, vae_speedup, custom_booster)
        # Return the loaded checkpoint (modelpatcher, clip, vae)
        return out
