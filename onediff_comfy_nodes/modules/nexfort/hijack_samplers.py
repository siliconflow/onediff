"""hijack ComfyUI/comfy/samplers.py
commit: 4bd7d55b9028d79829a645edfe8259f7b7a049c0
Date:   Thu Apr 11 22:43:05 2024 -0400
"""

from typing import Dict

import torch
from comfy.samplers import calc_cond_batch, can_concat_cond, cond_cat, get_area_and_mult

from ..sd_hijack_utils import Hijacker
from .booster_utils import is_using_nexfort_backend
from .patch_management import create_patch_executor, PatchType


def calc_cond_batch_of(orig_func, model, conds, x_in, timestep, model_options):
    out_conds = []
    out_counts = []
    to_run = []

    for i in range(len(conds)):
        out_conds.append(torch.zeros_like(x_in))
        out_counts.append(torch.ones_like(x_in) * 1e-37)

        cond = conds[i]
        if cond is not None:
            for x in cond:
                p = get_area_and_mult(x, x_in, timestep)
                if p is None:
                    continue

                to_run += [(p, i)]

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        # to_batch = to_batch_temp[:1]
        to_batch = to_batch_temp
        # free_memory = model_management.get_free_memory(x_in.device)
        # for i in range(1, len(to_batch_temp) + 1):
        #     batch_amount = to_batch_temp[:len(to_batch_temp)//i]
        #     input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
        #     if model.memory_required(input_shape) < free_memory:
        #         to_batch = batch_amount
        #         break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x)
        c = cond_cat(c)
        timestep_ = torch.cat([timestep] * batch_chunks)

        if control is not None:
            c["control"] = control.get_control(
                input_x, timestep_, c, len(cond_or_uncond)
            )

        transformer_options = {}
        if "transformer_options" in model_options:
            transformer_options = model_options["transformer_options"].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
                transformer_options["patches"] = cur_patches
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]

        diff_model = model.diffusion_model
        transformer_options["sigmas"] = timestep

        if create_patch_executor(PatchType.CachedCrossAttentionPatch).check_patch(
            diff_model
        ):
            patch_executor = create_patch_executor(PatchType.UNetExtraInputOptions)
            extra_options = transformer_options
            sigma = (
                extra_options["sigmas"][0].item()
                if "sigmas" in extra_options
                else 999999999.9
            )
            assert "_sigmas" not in extra_options
            extra_options["_sigmas"] = {}
            attn2_patch_dict = extra_options["patches_replace"]["attn2"]
            for k, attn_m in attn2_patch_dict.items():
                out_lst = []
                for i, callback in enumerate(attn_m.callback):
                    if (
                        sigma <= attn_m.kwargs[i]["sigma_start"]
                        and sigma >= attn_m.kwargs[i]["sigma_end"]
                    ):
                        out_lst.append(1)
                    else:
                        out_lst.append(0)
                # extra inputs
                transformer_options["_sigmas"][attn_m.forward_patch_key] = torch.randn(
                    *out_lst
                )

            # extra inputs
            transformer_options["_attn2"] = patch_executor.get_patch(diff_model)[
                "attn2"
            ]

        c["transformer_options"] = transformer_options
        if "model_function_wrapper" in model_options:
            output = model_options["model_function_wrapper"](
                model.apply_model,
                {
                    "input": input_x,
                    "timestep": timestep_,
                    "c": c,
                    "cond_or_uncond": cond_or_uncond,
                },
            ).chunk(batch_chunks)
        else:
            output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)

        for o in range(batch_chunks):
            cond_index = cond_or_uncond[o]
            a = area[o]
            if a is None:
                out_conds[cond_index] += output[o] * mult[o]
                out_counts[cond_index] += mult[o]
            else:
                out_c = out_conds[cond_index]
                out_cts = out_counts[cond_index]
                dims = len(a) // 2
                for i in range(dims):
                    out_c = out_c.narrow(i + 2, a[i + dims], a[i])
                    out_cts = out_cts.narrow(i + 2, a[i + dims], a[i])
                out_c += output[o] * mult[o]
                out_cts += mult[o]

    for i in range(len(out_conds)):
        out_conds[i] /= out_counts[i]

    return out_conds

    #     for o in range(batch_chunks):
    #         cond_index = cond_or_uncond[o]
    #         out_conds[cond_index][
    #             :,
    #             :,
    #             area[o][2] : area[o][0] + area[o][2],
    #             area[o][3] : area[o][1] + area[o][3],
    #         ] += (output[o] * mult[o])
    #         out_counts[cond_index][
    #             :,
    #             :,
    #             area[o][2] : area[o][0] + area[o][2],
    #             area[o][3] : area[o][1] + area[o][3],
    #         ] += mult[o]

    # for i in range(len(out_conds)):
    #     out_conds[i] /= out_counts[i]

    # return out_conds


def cond_func(orig_func, model, *args, **kwargs):
    return is_using_nexfort_backend(model)


samplers_hijack = Hijacker()
samplers_hijack.register(
    orig_func=calc_cond_batch,
    sub_func=calc_cond_batch_of,
    cond_func=cond_func,
)
