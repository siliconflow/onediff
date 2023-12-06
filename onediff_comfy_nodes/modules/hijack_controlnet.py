import comfy
import oneflow as flow
from packaging.version import Version
from comfy.controlnet import ControlLoraOps, ControlNet
from onediff.infer_compiler import oneflow_compile
from .sd_hijack_utils import Hijacker


def set_attr_of(obj, attr, value):
    if getattr(obj._deployable_module_model, "_oneflow_module", None) is not None:
        obj = obj._deployable_module_model._oneflow_module
        value = flow.utils.tensor.from_torch(value)
        attrs = attr.split(".")
        for name in attrs[:-1]:
            obj = getattr(obj, name)
        prev = getattr(obj, attrs[-1])
        prev.copy_(value)
    else:
        comfy.utils.set_attr(obj, attr, value)


class HijackControlLora:
    oneflow_model = None

    @staticmethod
    def cleanup():
        if HijackControlLora.oneflow_model is not None:
            del HijackControlLora.oneflow_model
            setattr(HijackControlLora, "oneflow_model", None)

    @staticmethod
    def pre_run(orig_func, self, model, percent_to_timestep_function):
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

    @staticmethod
    def cond_func(orig_func, *args, **kwargs):
        return True


controlnet_hijacker = Hijacker()
controlnet_hijacker.register(
    orig_func="comfy.controlnet.ControlLora.pre_run",
    sub_func=HijackControlLora.pre_run,
    cond_func=HijackControlLora.cond_func,
)

controlnet_hijacker.extend_unhijack(HijackControlLora.cleanup)
