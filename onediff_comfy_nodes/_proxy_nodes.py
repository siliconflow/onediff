from onediff.infer_compiler.with_proxy import Proxy

# ComfyUI
import folder_paths
from nodes import CheckpointLoaderSimple


class OneDiffCheckpointLoaderSimple_Proxy(CheckpointLoaderSimple):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_speedup": (["disable", "enable"],),
            }
        }

    CATEGORY = "OneDiff/Loaders"
    FUNCTION = "onediff_load_checkpoint"

    def onediff_load_checkpoint(
        self, ckpt_name, vae_speedup, output_vae=True, output_clip=True
    ):
        # CheckpointLoaderSimple.load_checkpoint
        modelpatcher, clip, vae = self.load_checkpoint(
            ckpt_name, output_vae, output_clip
        )

        modelpatcher.model.diffusion_model = Proxy(modelpatcher.model.diffusion_model)

        return modelpatcher, clip, vae
