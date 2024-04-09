"""hijack ComfyUI/comfy/samplers.py"""
from functools import total_ordering

import torch
from comfy.samplers import (
    calc_cond_uncond_batch,
    can_concat_cond,
    cond_cat,
    get_area_and_mult,
)

from onediff.infer_compiler.deployable_module import DeployableModule

from .sd_hijack_utils import Hijacker


def new_calc_cond_uncond_batch(
    orig_func, model, cond, uncond, x_in, timestep, model_options
):
    out_cond = torch.zeros_like(x_in)
    out_count = torch.ones_like(x_in) * 1e-37

    out_uncond = torch.zeros_like(x_in)
    out_uncond_count = torch.ones_like(x_in) * 1e-37

    COND = 0
    UNCOND = 1

    to_run = []
    for x in cond:
        p = get_area_and_mult(x, x_in, timestep)
        if p is None:
            continue

        to_run += [(p, COND)]
    if uncond is not None:
        for x in uncond:
            p = get_area_and_mult(x, x_in, timestep)
            if p is None:
                continue

            to_run += [(p, UNCOND)]

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
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        # transformer_options["sigmas"] = timestep
        if len(timestep) == 1:
            transformer_options["sigmas"] = timestep.item()
        else:
            transformer_options["sigmas"] = timestep

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
        del input_x

        for o in range(batch_chunks):
            if cond_or_uncond[o] == COND:
                out_cond[
                    :,
                    :,
                    area[o][2] : area[o][0] + area[o][2],
                    area[o][3] : area[o][1] + area[o][3],
                ] += (output[o] * mult[o])
                out_count[
                    :,
                    :,
                    area[o][2] : area[o][0] + area[o][2],
                    area[o][3] : area[o][1] + area[o][3],
                ] += mult[o]
            else:
                out_uncond[
                    :,
                    :,
                    area[o][2] : area[o][0] + area[o][2],
                    area[o][3] : area[o][1] + area[o][3],
                ] += (output[o] * mult[o])
                out_uncond_count[
                    :,
                    :,
                    area[o][2] : area[o][0] + area[o][2],
                    area[o][3] : area[o][1] + area[o][3],
                ] += mult[o]
        del mult

    out_cond /= out_count
    del out_count
    out_uncond /= out_uncond_count
    del out_uncond_count
    return out_cond, out_uncond


def cond_func(orig_func, model, *args, **kwargs):
    diff_model = model.diffusion_model
    if isinstance(diff_model, DeployableModule):
        return True
    else:
        return False


samplers_hijack = Hijacker()
samplers_hijack.register(
    orig_func=calc_cond_uncond_batch,
    sub_func=new_calc_cond_uncond_batch,
    cond_func=cond_func,
)
