import copy

import torch
import comfy


def state_dict_hook(module, state_dict, prefix, local_metadata):
    new_state_dict = type(state_dict)()
    for k, v in state_dict.items():
        # diffusion_model._deployable_module_model._torch_module.out.2.weight => diffusion_model.out.2.weight
        if k.startswith("diffusion_model._deployable_module_model"):
            x = k.split(".")
            new_k = ".".join(x[:1] + x[3:])
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


class OneFlowSpeedUpModelPatcher(comfy.model_patcher.ModelPatcher):
    def __init__(
        self,
        model,
        load_device,
        offload_device,
        size=0,
        current_device=None,
        weight_inplace_update=False,
        *,
        use_graph=None,
    ):
        from onediff.infer_compiler import oneflow_compile
        from onediff.infer_compiler.with_oneflow_compile import DeployableModule

        self.weight_inplace_update = weight_inplace_update
        self.object_patches = {}
        self.object_patches_backup = {}
        self.size = size
        self.model = copy.copy(model)
        self.model.__dict__["_modules"] = copy.copy(model.__dict__["_modules"])
        if isinstance(self.model.diffusion_model, DeployableModule):
            self.model.__dict__["_modules"][
                "diffusion_model"
            ] = self.model.diffusion_model
        else:
            self.model.__dict__["_modules"]["diffusion_model"] = oneflow_compile(
                self.model.diffusion_model, use_graph=use_graph
            )
        self.model._register_state_dict_hook(state_dict_hook)
        self.patches = {}
        self.backup = {}
        self.model_options = {"transformer_options": {}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device
        if current_device is None:
            self.current_device = self.offload_device
        else:
            self.current_device = current_device

    def clone(self):
        n = OneFlowSpeedUpModelPatcher(
            self.model,
            self.load_device,
            self.offload_device,
            self.size,
            self.current_device,
            weight_inplace_update=self.weight_inplace_update,
        )
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.model_keys = self.model_keys
        return n
