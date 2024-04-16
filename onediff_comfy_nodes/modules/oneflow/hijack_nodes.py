from nodes import ControlNetApply, ControlNetApplyAdvanced

from ..sd_hijack_utils import Hijacker
from .onediff_controlnet import OneDiffControlLora


def apply_controlnet_base(orig_func, self, conditioning, control_net, image, strength):
    if strength == 0:
        return (conditioning,)

    c = []
    control_hint = image.movedim(-1, 1)
    for t in conditioning:
        n = [t[0], t[1].copy()]
        if not hasattr(self, "_c_net"):
            self._c_net = control_net.copy()

        c_net = self._c_net.set_cond_hint(control_hint, strength)

        if "control" in t[1]:
            c_net.set_previous_controlnet(t[1]["control"])
        n[1]["control"] = c_net
        n[1]["control_apply_to_uncond"] = True
        c.append(n)
    return (c,)


def apply_controlnet_cond_func_base(
    orig_func, self, conditioning, control_net, image, strength
):
    return isinstance(control_net, OneDiffControlLora)


def apply_controlnet_advanced(
    orig_func,
    self,
    positive,
    negative,
    control_net,
    image,
    strength,
    start_percent,
    end_percent,
):
    if strength == 0:
        return (positive, negative)
    control_hint = image.movedim(-1, 1)
    cnets = {}

    out = []
    for conditioning in [positive, negative]:
        c = []
        for t in conditioning:
            d = t[1].copy()

            prev_cnet = d.get("control", None)
            if prev_cnet in cnets:
                c_net = cnets[prev_cnet]
            else:
                if not hasattr(self, "_c_net"):
                    print("creating new cnet")
                    self._c_net = control_net.copy()
                c_net = self._c_net.set_cond_hint(
                    control_hint, strength, (start_percent, end_percent)
                )
                c_net.set_previous_controlnet(prev_cnet)
                cnets[prev_cnet] = c_net

            d["control"] = c_net
            d["control_apply_to_uncond"] = False
            n = [t[0], d]
            c.append(n)
        out.append(c)
    return (out[0], out[1])


def apply_controlnet_cond_func_advanced(
    orig_func,
    self,
    positive,
    negative,
    control_net,
    image,
    strength,
    start_percent,
    end_percent,
):
    return isinstance(control_net, OneDiffControlLora)


nodes_hijacker = Hijacker()
nodes_hijacker.register(
    orig_func=ControlNetApply.apply_controlnet,
    sub_func=apply_controlnet_base,
    cond_func=apply_controlnet_cond_func_base,
)
nodes_hijacker.register(
    orig_func=ControlNetApplyAdvanced.apply_controlnet,
    sub_func=apply_controlnet_advanced,
    cond_func=apply_controlnet_cond_func_advanced,
)
