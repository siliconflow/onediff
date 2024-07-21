import inspect

import comfy
from comfy.controlnet import ControlLora, ControlLoraOps, ControlNet


class OneDiffControlLora(ControlLora):
    @classmethod
    def from_controllora(cls, controlnet: ControlLora, *, compile_fn: callable = None):
        init_parameters = set(inspect.signature(cls.__init__).parameters.keys())
        init_dict = {
            attr: getattr(controlnet, attr)
            for attr in init_parameters
            if attr != "self"
        }
        c = cls(**init_dict)
        controlnet.copy_to(c)
        c._control_model = None
        c._compile_fn = compile_fn
        return c

    def pre_run(self, model, percent_to_timestep_function):
        ControlNet.pre_run(self, model, percent_to_timestep_function)

        self.manual_cast_dtype = model.manual_cast_dtype
        dtype = model.get_dtype()
        if self.manual_cast_dtype is None:

            class control_lora_ops(ControlLoraOps, comfy.ops.disable_weight_init):
                pass

        else:

            class control_lora_ops(ControlLoraOps, comfy.ops.manual_cast):
                pass

            dtype = self.manual_cast_dtype
        if self._control_model is None:
            controlnet_config = model.model_config.unet_config.copy()
            controlnet_config.pop("out_channels")
            controlnet_config["hint_channels"] = self.control_weights[
                "input_hint_block.0.weight"
            ].shape[1]
            controlnet_config["operations"] = control_lora_ops
            controlnet_config["dtype"] = dtype
            self.control_model = comfy.cldm.cldm.ControlNet(**controlnet_config)
            self.control_model.to(comfy.model_management.get_torch_device())
            self._control_model = self._compile_fn(self.control_model)

        self.control_model = self._control_model
        diffusion_model = model.diffusion_model
        sd = diffusion_model.state_dict()
        # cm = self.control_model.state_dict()

        for k in sd:
            weight = sd[k]
            try:
                comfy.utils.set_attr_param(self.control_model, k, weight)
            except:
                pass

        for k in self.control_weights:
            if k not in {"lora_controlnet"}:
                comfy.utils.set_attr_param(
                    self.control_model,
                    k,
                    self.control_weights[k]
                    .to(dtype)
                    .to(comfy.model_management.get_torch_device()),
                )

    def cleanup(self):
        pass

    def copy(self):
        init_parameters = set(inspect.signature(type(self).__init__).parameters.keys())
        init_dict = {
            attr: getattr(self, attr) for attr in init_parameters if attr != "self"
        }
        c = type(self)(**init_dict)

        self.copy_to(c)
        c._control_model = self._control_model
        c._compile_fn = self._compile_fn
        return c
