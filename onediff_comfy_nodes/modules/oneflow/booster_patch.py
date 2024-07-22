import os
from functools import singledispatchmethod

from comfy.controlnet import ControlLora, ControlNet
from comfy.model_patcher import ModelPatcher

from comfy.sd import VAE
from onediff.infer_compiler.backends.oneflow import (
    OneflowDeployableModule as DeployableModule,
)

from ..booster_interface import BoosterExecutor


class PatchBoosterExecutor(BoosterExecutor):
    @singledispatchmethod
    def execute(self, model, ckpt_name=None):
        return model

    def _set_batch_size_patch(self, diff_model: DeployableModule, latent_image):
        batch_size = latent_image["samples"].shape[0]
        if isinstance(diff_model, DeployableModule):
            file_path = diff_model.get_graph_file()
            if file_path is None:
                return diff_model

            file_dir = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            names = file_name.split("_")
            key, is_replace = "bs=", False
            for i, name in enumerate(names):
                if key in name:
                    names[i] = f"{key}{batch_size}"
                    is_replace = True
            if not is_replace:
                names = [f"{key}{batch_size}"] + names

            new_file_name = "_".join(names)
            new_file_path = os.path.join(file_dir, new_file_name)

            diff_model.set_graph_file(new_file_path)
        else:
            print(f"Warning: model is not a {DeployableModule}")
        return diff_model

    @execute.register(ModelPatcher)
    def _(self, model, **kwargs):
        latent_image = kwargs.get("latent_image", None)
        if latent_image:
            diff_model = model.model.diffusion_model
            self._set_batch_size_patch(diff_model, latent_image)
        return model
