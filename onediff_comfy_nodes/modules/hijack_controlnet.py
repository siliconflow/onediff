import comfy
import oneflow as flow
from comfy.controlnet import ControlLoraOps, ControlNet, ControlLora
from onediff.infer_compiler import oneflow_compile


def set_attr_of(obj, attr, value, *, debug=True):
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
    if debug and exist_oneflow_module:
        fail_count = 0
        try:
            _set_attr_of(obj, attr, value)
        except:
            fail_count += 1

        try:
            obj = obj._deployable_module_model._torch_module
            comfy.utils.set_attr(obj, attr, value)
        except:
            fail_count += 1

        if fail_count == 1:
            raise Exception(f"set_attr_of failed {type(obj)} {attr} {type(value)}")
        return

    if exist_oneflow_module:
        _set_attr_of(obj, attr, value)
    else:
        comfy.utils.set_attr(obj, attr, value)


class HijackControlNet(ControlNet):
    @classmethod
    def from_controlnet(cls, controlnet: ControlNet):
        c = cls(
            controlnet.control_model,
            global_average_pooling=controlnet.global_average_pooling,
        )
        c.cond_hint_original = controlnet.cond_hint_original
        c.strength = controlnet.strength
        c.timestep_percent_range = controlnet.timestep_percent_range
        return c

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        # compile to oneflow
        self.control_model = oneflow_compile(self.control_model)

    def copy(self):
        c = HijackControlNet(
            self.control_model, global_average_pooling=self.global_average_pooling
        )
        self.copy_to(c)
        return c


class HijackControlLora(ControlLora):
    oneflow_model = None

    @classmethod
    def from_controllora(cls, controlnet: ControlLora):
        c = cls(
            controlnet.control_weights,
            global_average_pooling=controlnet.global_average_pooling,
            device=controlnet.device,
        )
        c.cond_hint_original = controlnet.cond_hint_original
        c.strength = controlnet.strength
        c.timestep_percent_range = controlnet.timestep_percent_range
        return c

    def pre_run(self, model, percent_to_timestep_function):
        # print("Hijacking ControlLora.pre_run")
        dtype = model.get_dtype()
        # super().pre_run(model, percent_to_timestep_function)
        ControlNet.pre_run(self, model, percent_to_timestep_function)

        if HijackControlLora.oneflow_model is None:
            controlnet_config = model.model_config.unet_config.copy()
            controlnet_config.pop("out_channels")
            controlnet_config["hint_channels"] = self.control_weights[
                "input_hint_block.0.weight"
            ].shape[1]
            controlnet_config["operations"] = ControlLoraOps()
            self.control_model = comfy.cldm.cldm.ControlNet(**controlnet_config)
            self.control_model.to(dtype)
            self.control_model.to(comfy.model_management.get_torch_device())
            HijackControlLora.oneflow_model = oneflow_compile(self.control_model)

        self.control_model = HijackControlLora.oneflow_model

        diffusion_model = model.diffusion_model
        sd = diffusion_model.state_dict()
        cm = self.control_model.state_dict()
        for k in sd:
            weight = comfy.model_management.resolve_lowvram_weight(
                sd[k], diffusion_model, k
            )
            try:
                set_attr_of(self.control_model, k, weight)
            except:
                pass

        for k in self.control_weights:
            if k not in {"lora_controlnet"}:
                weight = (
                    self.control_weights[k]
                    .to(dtype)
                    .to(comfy.model_management.get_torch_device())
                )
                set_attr_of(self.control_model, k, weight)

        lazy_loader = getattr(HijackControlLora, "lazy_load_hook", None)
        if lazy_loader and callable(lazy_loader):
            lazy_loader(HijackControlLora.oneflow_model)
            delattr(HijackControlLora, "lazy_load_hook")

    def copy(self):
        c = HijackControlLora(
            self.control_weights, global_average_pooling=self.global_average_pooling
        )
        self.copy_to(c)
        return c
