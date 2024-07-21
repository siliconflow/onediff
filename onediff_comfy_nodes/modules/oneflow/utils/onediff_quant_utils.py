import os

import comfy
import torch
import torch.nn as nn

if hasattr(comfy.ops, "disable_weight_init"):
    comfy_ops_Linear = comfy.ops.disable_weight_init.Linear
else:
    print(
        "Warning: ComfyUI version is too old, please upgrade it. github: git@github.com:comfyanonymous/ComfyUI.git "
    )
    comfy_ops_Linear = comfy.ops.Linear

__all__ = ["replace_module_with_quantizable_module"]


def get_sub_module(module, sub_module_name) -> nn.Module:
    """Get a submodule of a module using dot-separated names.

    Args:
        module (nn.Module): The base module.
        sub_module_name (str): Dot-separated name of the submodule.

    Returns:
        nn.Module: The requested submodule.
    """

    parts = sub_module_name.split(".")
    current_module = module

    for part in parts:
        try:
            if part.isdigit():
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
        except (IndexError, AttributeError):
            raise ModuleNotFoundError(f"Submodule {part} not found.")

    return current_module


def modify_sub_module(module, sub_module_name, new_value):
    """Modify a submodule of a module using dot-separated names.

    Args:
        module (nn.Module): The base module.
        sub_module_name (str): Dot-separated name of the submodule.
        new_value: The new value to assign to the submodule.

    """
    parts = sub_module_name.split(".")
    current_module = module

    for i, part in enumerate(parts):
        try:
            if part.isdigit():
                if i == len(parts) - 1:
                    current_module[int(part)] = new_value
                else:
                    current_module = current_module[int(part)]
            else:
                if i == len(parts) - 1:
                    setattr(current_module, part, new_value)
                else:
                    current_module = getattr(current_module, part)
        except (IndexError, AttributeError):
            raise ModuleNotFoundError(f"Submodule {part} not found.")


def _load_calibrate_info(calibrate_info_path):
    calibrate_info = {}
    with open(calibrate_info_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            items = line.split(" ")
            calibrate_info[items[0]] = [
                float(items[1]),
                int(items[2]),
                [float(x) for x in items[3].split(",")],
            ]
    return calibrate_info


def search_modules(root, match_fn: callable, name=""):
    """
    example:
    >>> search_modules(model, lambda m: isinstance(m, (nn.Conv2d, nn.Linear))
    """
    if match_fn(root):
        return {name: root}

    result = {}
    for child_name, child in root.named_children():
        result.update(
            search_modules(
                child, match_fn, f"{name}.{child_name}" if name != "" else child_name
            )
        )
    return result


def _can_use_flash_attn(attn):
    dim_head = attn.to_q.out_features // attn.heads
    if dim_head != 40 and dim_head != 64:
        return False
    if attn.to_k is None or attn.to_v is None:
        return False
    if (
        attn.to_q.bias is not None
        or attn.to_k.bias is not None
        or attn.to_v.bias is not None
    ):
        return False
    if (
        attn.to_q.in_features != attn.to_k.in_features
        or attn.to_q.in_features != attn.to_v.in_features
    ):
        return False
    if not (
        attn.to_q.weight.dtype == attn.to_k.weight.dtype
        and attn.to_q.weight.dtype == attn.to_v.weight.dtype
    ):
        return False
    return True


def _rewrite_attention(attn):
    from onediff_quant.models import DynamicQuantLinearModule, StaticQuantLinearModule

    dim_head = attn.to_q.out_features // attn.heads
    has_bias = attn.to_q.bias is not None
    attn.to_qkv = nn.Linear(
        attn.to_q.in_features, attn.to_q.out_features * 3, bias=has_bias
    )
    attn.to_qkv.requires_grad_(False)
    qkv_weight = torch.cat(
        [
            attn.to_q.weight.permute(1, 0).reshape(-1, attn.heads, dim_head),
            attn.to_k.weight.permute(1, 0).reshape(-1, attn.heads, dim_head),
            attn.to_v.weight.permute(1, 0).reshape(-1, attn.heads, dim_head),
        ],
        dim=2,
    )
    qkv_weight = (
        qkv_weight.reshape(-1, attn.to_q.out_features * 3).permute(1, 0).contiguous()
    )
    attn.to_qkv.weight.data = qkv_weight

    if has_bias:
        qkv_bias = (
            torch.cat(
                [
                    attn.to_q.bias.reshape(attn.heads, dim_head),
                    attn.to_k.bias.reshape(attn.heads, dim_head),
                    attn.to_v.bias.reshape(attn.heads, dim_head),
                ],
                dim=1,
            )
            .reshape(attn.to_q.out_features * 3)
            .contiguous()
        )
        attn.to_qkv.bias.data = qkv_bias

    if isinstance(attn.to_q, StaticQuantLinearModule) or isinstance(
        attn.to_q, DynamicQuantLinearModule
    ):
        cls = type(attn.to_q)
        weight_scale = (
            torch.cat(
                [
                    torch.Tensor(attn.to_q.calibrate[2]).reshape(attn.heads, dim_head),
                    torch.Tensor(attn.to_k.calibrate[2]).reshape(attn.heads, dim_head),
                    torch.Tensor(attn.to_v.calibrate[2]).reshape(attn.heads, dim_head),
                ],
                dim=1,
            )
            .reshape(attn.to_q.out_features * 3)
            .contiguous()
        )
        calibrate = [attn.to_q.calibrate[0], attn.to_q.calibrate[1], weight_scale]

        old_env = os.getenv("ONEFLOW_FUSE_QUANT_TO_MATMUL")
        os.environ["ONEFLOW_FUSE_QUANT_TO_MATMUL"] = "0"
        attn.to_qkv = cls(attn.to_qkv, attn.to_q.nbits, calibrate, attn.to_q.name)
        attn.scale = dim_head**-0.5

        os.environ["ONEFLOW_FUSE_QUANT_TO_MATMUL"] = old_env


def replace_module_with_quantizable_module(
    diffusion_model, calibrate_info_path, use_rewrite_attn=True
):
    from onediff_quant.utils import get_quantize_module

    calibrate_info = _load_calibrate_info(calibrate_info_path)
    for sub_module_name, sub_calibrate_info in calibrate_info.items():
        sub_mod = get_sub_module(diffusion_model, sub_module_name)

        if isinstance(sub_mod, comfy_ops_Linear):
            # fix onediff_quant use isinstance(sub_mod, torch.nn.Linear)
            sub_mod.__class__ = torch.nn.Linear

        sub_mod.weight.requires_grad = False
        sub_mod.weight.data = sub_mod.weight.to(torch.int8)
        sub_mod.cuda()  # TODO: remove this line , because we onediff_quant pkg weight_scale
        sub_mod = get_quantize_module(
            sub_mod,
            sub_module_name,
            sub_calibrate_info,
            fake_quant=False,
            static=False,
            nbits=8,
        )
        modify_sub_module(diffusion_model, sub_module_name, sub_mod)
    if use_rewrite_attn:
        print(f"{use_rewrite_attn=}, rewrite CrossAttention")
        try:
            # rewrite CrossAttention to use qkv
            from comfy.ldm.modules.attention import CrossAttention

            match_func = lambda m: isinstance(
                m, CrossAttention
            ) and _can_use_flash_attn(m)
            can_rewrite_modules = search_modules(diffusion_model, match_func)
            print(f"rewrite {len(can_rewrite_modules)=} CrossAttention")
            for k, v in can_rewrite_modules.items():
                if f"{k}.to_q" in calibrate_info:
                    _rewrite_attention(v)  # diffusion_model is modified in-place
                else:
                    print(f"skip {k+'.to_q'} not in calibrate_info")

        except Exception as e:
            raise RuntimeError(f"rewrite CrossAttention failed: {e}")


def find_quantizable_modules(
    module, name="", *, quantize_conv=True, quantize_linear=True
):
    if isinstance(module, nn.Conv2d) and quantize_conv:
        return {name: module}

    if isinstance(module, nn.Linear) and quantize_linear:
        return {name: module}

    res = {}
    for child_name, child in module.named_children():
        res.update(
            find_quantizable_modules(
                child,
                name + "." + child_name if name != "" else child_name,
                quantize_conv=quantize_conv,
                quantize_linear=quantize_linear,
            )
        )
    return res


def quantize_and_save_model(
    diffusion_model,
    output_dir,
    *,
    quantize_conv=True,
    quantize_linear=True,
    verbose=False,
    bits=8,
):
    import time

    from onediff_quant import Quantizer
    from onediff_quant.utils import symm_quantize_sub_module
    from safetensors.torch import save_model

    print(
        f"quantize and save_model, conv={quantize_conv}, linear={quantize_linear}, verbose={verbose}, output={output_dir}"
    )
    start_time = time.time()

    print("Find the quantizable modules...")
    quantizable_modules = find_quantizable_modules(
        diffusion_model, quantize_conv=quantize_conv, quantize_linear=quantize_linear
    )
    print(f"quantizable_modules size: {len(quantizable_modules)}")

    if verbose:
        for name, module in quantizable_modules.items():
            print(name, ": ", module)

    calibrate_info = {}

    enum_quantizable_modules = enumerate(quantizable_modules.items())
    _quant_total = len(quantizable_modules.items())
    for i, (name, module) in enum_quantizable_modules:
        if verbose:
            print(f"Calculate quantization infos of {name} ...")
        quantizer = Quantizer()
        quantizer.configure(bits=bits, perchannel=True)
        quantizer.find_params(module.weight.float(), weight=True)
        shape = [-1] + [1] * (len(module.weight.shape) - 1)
        scale = quantizer.scale.reshape(*shape)
        symm_quantize_sub_module(diffusion_model, name, scale, quantizer.maxq)
        calibrate_info[name] = [scale]

    print(f"""save quantized model to {output_dir}""")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise RuntimeError(
            f"{os.path.basename(output_dir)} has existed, rename the out_dir and try again."
        )
    save_model(diffusion_model, os.path.join(output_dir, "unet_int8.safetensors"))

    print(f'save calibrate_info to {os.path.join(output_dir, "calibrate_info.txt")}')

    with open(os.path.join(output_dir, "calibrate_info.txt"), "w") as f:
        for name, info in calibrate_info.items():
            input_scale = 0
            input_zero_point = 0
            weight_scale = [str(x) for x in info[0].reshape(-1).tolist()]
            f.write(
                f"{name} {input_scale} {input_zero_point} {','.join(weight_scale)}\n"
            )

    print(f"Quantize module time: {time.time() - start_time}s")
