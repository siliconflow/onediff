import folder_paths
import torch
from nodes import CheckpointLoaderSimple, ControlNetLoader

from .modules.optimizer_scheduler import OptimizerScheduler
from .utils.import_utils import is_nexfort_available  # type: ignore
from .utils.import_utils import is_oneflow_available

if is_oneflow_available():
    from .modules.oneflow.optimizer_basic import BasicOneFlowOptimizerExecutor
    BasicOptimizerExecutor = BasicOneFlowOptimizerExecutor
elif is_nexfort_available():
    pass 
else:
    raise RuntimeError()

__all__ = [
    "ModelSpeedup",
    "VaeSpeedup",
    "ControlnetSpeedup",
    "OneDiffApplyModelOptimizer"
    "OneDiffControlNetLoader",
    "OneDiffControlNetLoader",
]

class ModelSpeedup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"model": ("MODEL",), "inplace": ([False, True],),},
            "optional": {
                "custom_optimizer": ("CUSTOM_OPTIMIZER",),
            }

        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "speedup"
    CATEGORY = "OneDiff"

    @torch.no_grad()
    def speedup(self, model, inplace=False, custom_optimizer: OptimizerScheduler=None):
        if custom_optimizer:
            op = custom_optimizer
            op.inplace = False
        else:
            op = OptimizerScheduler(BasicOptimizerExecutor(), inplace=inplace)

        optimized_model = op.compile(model)
        return (optimized_model,)

class VaeSpeedup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"vae": ("VAE",),},
            "optional": {
                "custom_optimizer": ("CUSTOM_OPTIMIZER",),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "speedup"
    CATEGORY = "OneDiff"

    @torch.no_grad()
    def speedup(self, vae, custom_optimizer=None):
        if custom_optimizer:
            op = custom_optimizer
        else:
            op = OptimizerScheduler(BasicOptimizerExecutor())

        new_vae = op(vae)
        return (new_vae,)
    
class ControlnetSpeedup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "control_net": ("CONTROL_NET",),
                "cnet_stack": ("CONTROL_NET_STACK",),
                "custom_optimizer": ("CUSTOM_OPTIMIZER",),
            }
        }

    RETURN_TYPES = ("CONTROL_NET", "CONTROL_NET_STACK",)
    FUNCTION = "speedup"
    CATEGORY = "OneDiff"

    @torch.no_grad()
    def speedup(self,control_net=None, cnet_stack=[], custom_optimizer=None):
        if custom_optimizer:
            op = custom_optimizer
        else:
            op = OptimizerScheduler(BasicOptimizerExecutor(), inplace=True)

        if control_net:
            control_net = op.compile(control_net)

        new_cnet_stack =[]
        for cnet in cnet_stack:
            new_cnet = tuple([op.compile(cnet[0])]+list(cnet[1:]))
            new_cnet_stack.append(new_cnet)
        return (control_net, new_cnet_stack,)
    
class OneDiffApplyModelOptimizer:
    """Main class responsible for optimizing models."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "quantization_optimizer": ("QuantizationOptimizer",),
                "deepcache_optimizer": ("DeepCacheOptimizer",),
                "torchcompile_optimizer":("TorchCompileOptimizer",),
            },
        }

    CATEGORY = "OneDiff/Optimization"
    RETURN_TYPES = ("CUSTOM_OPTIMIZER",)
    FUNCTION = "optimize_model"

    @torch.no_grad()
    def optimize_model(self, quantization_optimizer=None, deepcache_optimizer=None, torchcompile_optimizer=None):
        """Apply the optimization technique to the model."""
        optimizers = []
        if quantization_optimizer:
            optimizers.append(quantization_optimizer)
        if deepcache_optimizer:
            optimizers.append(deepcache_optimizer)
        if torchcompile_optimizer:
            optimizers.append(torchcompile_optimizer)

        assert len(optimizers) > 0
        return (OptimizerScheduler(optimizers),)

class OneDiffControlNetLoader(ControlNetLoader):
    @classmethod
    def INPUT_TYPES(s):
        ret = super().INPUT_TYPES()
        ret.update({"optional": {
                    "model_optimizer": ("MODEL_OPTIMIZER",),}
        })
        return ret 

    CATEGORY = "OneDiff/Loaders"
    FUNCTION = "onediff_load_controlnet"

    @torch.no_grad()
    def onediff_load_controlnet(self, control_net_name, custom_optimizer=None):
        controlnet = super().load_controlnet(control_net_name)[0]
        op = OptimizerScheduler(BasicOptimizerExecutor())
        controlnet = op.compile(controlnet)
        return (controlnet,)

class OneDiffCheckpointLoaderSimple(CheckpointLoaderSimple):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_speedup": (["disable", "enable"],),
            },
            "optional": {
                    "custom_optimizer": ("CUSTOM_OPTIMIZER",),
            }
        }

    CATEGORY = "OneDiff/Loaders"
    FUNCTION = "onediff_load_checkpoint"

    @torch.no_grad()
    def onediff_load_checkpoint(
        self, ckpt_name, vae_speedup="disable", output_vae=True, output_clip=True, model_optimizer: OptimizerScheduler=None,
    ):
        # CheckpointLoaderSimple.load_checkpoint
        modelpatcher, clip, vae = self.load_checkpoint(
            ckpt_name, output_vae, output_clip
        )
        if model_optimizer is None:
            model_optimizer = OptimizerScheduler(BasicOptimizerExecutor())
        modelpatcher = model_optimizer.compile(modelpatcher, ckpt_name=ckpt_name)
        if vae_speedup == "enable":
            vae_optimizer = OptimizerScheduler(BasicOptimizerExecutor())
            vae = vae_optimizer.compile(vae, ckpt_name=ckpt_name)
        # set inplace update
        modelpatcher.weight_inplace_update = True
        return modelpatcher, clip, vae



