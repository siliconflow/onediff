import comfy
import oneflow as flow  # usort: skip
import torch
from comfy.controlnet import ControlLora, ControlLoraOps, ControlNet
from onediff.infer_compiler import oneflow_compile

__all__ = ["OneDiffControlLora"]


def set_attr_of(obj, attr, value):
    def _set_attr_of(obj, attr, value):
        obj = obj._deployable_module_model._oneflow_module
        value = flow.utils.tensor.from_torch(value)
        attrs = attr.split(".")
        for name in attrs[:-1]:
            obj = getattr(obj, name)
        prev = getattr(obj, attrs[-1])
        prev.copy_(value)

    exist_oneflow_module = (
        getattr(obj._deployable_module_model, "_oneflow_module", None) is not None
    )

    if exist_oneflow_module:
        _set_attr_of(obj, attr, value)
    else:
        attrs = attr.split(".")
        for name in attrs[:-1]:
            obj = getattr(obj, name)
        prev = getattr(obj, attrs[-1])
        setattr(obj, attrs[-1], torch.nn.Parameter(value, requires_grad=False))
        del prev


class OneDiffControlLora(ControlLora):
    @classmethod
    def from_controllora(
        cls, controlnet: ControlLora, *, gen_compile_options: callable = None
    ):
        c = cls(
            controlnet.control_weights,
            global_average_pooling=controlnet.global_average_pooling,
        )
        controlnet.copy_to(c)
        c._oneflow_model = None
        c.gen_compile_options = gen_compile_options
        return c

    def pre_run(self, model, percent_to_timestep_function):
        dtype = model.get_dtype()
        # super().pre_run(model, percent_to_timestep_function)
        ControlNet.pre_run(self, model, percent_to_timestep_function)
        self.manual_cast_dtype = model.manual_cast_dtype

        if self._oneflow_model is None:
            controlnet_config = model.model_config.unet_config.copy()
            controlnet_config.pop("out_channels")
            controlnet_config["hint_channels"] = self.control_weights[
                "input_hint_block.0.weight"
            ].shape[1]
            dtype = model.get_dtype()

            if self.manual_cast_dtype is None:

                class control_lora_ops(ControlLoraOps, comfy.ops.disable_weight_init):
                    pass

            else:

                class control_lora_ops(ControlLoraOps, comfy.ops.manual_cast):
                    pass

                dtype = self.manual_cast_dtype

            controlnet_config["operations"] = control_lora_ops()
            self.control_model = comfy.cldm.cldm.ControlNet(**controlnet_config)
            self.control_model.to(dtype)
            self.control_model.to(comfy.model_management.get_torch_device())

            file_device_dict = (
                self.gen_compile_options(self.control_model)
                if self.gen_compile_options is not None
                else {}
            )
            self._oneflow_model = oneflow_compile(self.control_model)
            compiled_options = self._oneflow_model._deployable_module_options
            compiled_options.graph_file = file_device_dict.get("graph_file", None)
            compiled_options.graph_file_device = file_device_dict.get(
                "graph_file_device", None
            )

        self.control_model = self._oneflow_model

        diffusion_model = model.diffusion_model
        sd = diffusion_model.state_dict()
        cm = self.control_model.state_dict()
        for k in sd:
            weight = comfy.model_management.resolve_lowvram_weight(
                sd[k], diffusion_model, k
            )
            try:
                set_attr_of(self.control_model, k, weight)
            except Exception as e:
                pass

        for k in self.control_weights:
            if k not in {"lora_controlnet"}:
                weight = (
                    self.control_weights[k]
                    .to(dtype)
                    .to(comfy.model_management.get_torch_device())
                )
                set_attr_of(self.control_model, k, weight)

    def copy(self):
        c = OneDiffControlLora(
            self.control_weights,
            global_average_pooling=self.global_average_pooling,
        )
        self.copy_to(c)
        c._oneflow_model = self._oneflow_model
        c.gen_compile_options = self.gen_compile_options
        return c
