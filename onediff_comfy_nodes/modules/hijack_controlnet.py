import torch
import comfy
import oneflow as flow
from packaging.version import Version
from comfy.controlnet import ControlLoraOps, ControlNet, ControlBase, broadcast_image_to
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


class HijackControlNet:
    @staticmethod
    def pre_run(orig_func, self, model, percent_to_timestep_function):
        orig_func(self, model, percent_to_timestep_function)
        self.control_model = oneflow_compile(self.control_model)


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


class HijackT2IAdapter:
    @staticmethod
    def get_control(orig_func, self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(
                x_noisy, t, cond, batched_number
            )

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        if (
            self.cond_hint is None
            or x_noisy.shape[2] * 8 != self.cond_hint.shape[2]
            or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]
        ):
            if self.cond_hint is not None:
                del self.cond_hint
            self.control_input = None
            self.cond_hint = None
            width, height = self.scale_image_to(
                x_noisy.shape[3] * 8, x_noisy.shape[2] * 8
            )
            self.cond_hint = (
                comfy.utils.common_upscale(
                    self.cond_hint_original, width, height, "nearest-exact", "center"
                )
                .float()
                .to(self.device)
            )
            if self.channels_in == 1 and self.cond_hint.shape[1] > 1:
                self.cond_hint = torch.mean(self.cond_hint, 1, keepdim=True)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(
                self.cond_hint, x_noisy.shape[0], batched_number
            )
        if self.control_input is None:
            # self.t2i_model.to(x_noisy.dtype)
            self.t2i_model = self.t2i_model.to(self.device)
            self.t2i_model = oneflow_compile(self.t2i_model)
            self.control_input = self.t2i_model(self.cond_hint.to(x_noisy.dtype))
            # self.t2i_model.cpu()

        control_input = list(
            map(lambda a: None if a is None else a.clone(), self.control_input)
        )
        mid = None
        if self.t2i_model.xl == True:
            mid = control_input[-1:]
            control_input = control_input[:-1]
        return self.control_merge(control_input, mid, control_prev, x_noisy.dtype)


controlnet_hijacker = Hijacker()
controlnet_hijacker.register(
    orig_func="comfy.controlnet.ControlLora.pre_run",
    sub_func=HijackControlLora.pre_run,
    cond_func=HijackControlLora.cond_func,
)


controlnet_hijacker.register(
    orig_func=comfy.controlnet.ControlNet.pre_run,
    sub_func=HijackControlNet.pre_run,
    cond_func=lambda orig_func, *args, **kwargs: True,
)


# TODO: fix this
# controlnet_hijacker.register(
#     orig_func=comfy.controlnet.T2IAdapter.get_control,
#     sub_func=HijackT2IAdapter.get_control,
#     cond_func=lambda orig_func, *args, **kwargs: True,
# )
controlnet_hijacker.extend_unhijack(HijackControlLora.cleanup)
