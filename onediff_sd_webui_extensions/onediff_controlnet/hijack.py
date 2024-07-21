import onediff_shared
import torch
from compile import is_oneflow_backend
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from modules.sd_hijack_utils import CondFunc
from onediff_utils import check_structure_change, singleton_decorator

from .utils import get_controlnet_script


def hijacked_main_entry(self, p):
    self._original_controlnet_main_entry(p)
    from .compile import compile_controlnet_ldm_unet
    from .model import OnediffControlNetModel

    if not onediff_shared.onediff_enabled:
        return
    sd_ldm = p.sd_model
    unet = sd_ldm.model.diffusion_model

    structure_changed = check_structure_change(
        onediff_shared.previous_unet_type, sd_ldm
    )
    if not onediff_shared.controlnet_compiled or structure_changed:
        onediff_model = OnediffControlNetModel(unet)
        onediff_shared.current_unet_graph = compile_controlnet_ldm_unet(
            sd_ldm, onediff_model
        )
        onediff_shared.controlnet_compiled = True


# When OneDiff is initializing, the controlnet extension has not yet been loaded.
# Therefore, this function should be called during image generation
# rather than during the initialization of the OneDiff.
@singleton_decorator
def hijack_controlnet_extension(p):
    CondFunc(
        "scripts.hook.UnetHook.hook",
        hijacked_controlnet_hook,
        lambda _, *arg, **kwargs: onediff_shared.onediff_enabled,
    )
    # get controlnet script
    controlnet_script = get_controlnet_script(p)
    if controlnet_script is None:
        return

    controlnet_script._original_controlnet_main_entry = (
        controlnet_script.controlnet_main_entry
    )
    controlnet_script.controlnet_main_entry = hijacked_main_entry.__get__(
        controlnet_script
    )


def unhijack_controlnet_extension(p):
    controlnet_script = get_controlnet_script(p)
    if controlnet_script is None:
        return

    if hasattr(controlnet_script, "_original_controlnet_main_entry"):
        controlnet_script.controlnet_main_entry = (
            controlnet_script._original_controlnet_main_entry
        )
        delattr(controlnet_script, "_original_controlnet_main_entry")


# We were intended to only hack the closure function `forward`
# in the member function `hook` of the UnetHook class in the ControlNet extension.
# But due to certain limitations, we were unable to directly only hack
# the closure function `forward` within the `hook` method.
# So we have to hack the entire member function `hook` in the `UnetHook` class.

# The function largely retains its original content,
# with modifications specifically made within the `forward` function.
# To identify the altered parts, you can search for the tag "modified by OneDiff"

# https://github.com/Mikubill/sd-webui-controlnet/blob/8bbbd0e55ef6e5d71b09c2de2727b36e7bc825b0/scripts/hook.py#L442
def hijacked_controlnet_hook(
    orig_func,
    self,
    model,
    sd_ldm,
    control_params,
    process,
    batch_option_uint_separate=False,
    batch_option_style_align=False,
):
    from modules import devices, lowvram, scripts, shared
    from scripts.controlnet_sparsectrl import SparseCtrl
    from scripts.enums import AutoMachine, ControlModelType, HiResFixOption
    from scripts.hook import (
        AbstractLowScaleModel,
        blur,
        mark_prompt_context,
        predict_noise_from_start,
        predict_q_sample,
        predict_start_from_noise,
        register_schedule,
        torch_dfs,
        unmark_prompt_context,
    )
    from scripts.ipadapter.ipadapter_model import ImageEmbed
    from scripts.logging import logger

    self.model = model
    self.sd_ldm = sd_ldm
    self.control_params = control_params

    model_is_sdxl = getattr(self.sd_ldm, "is_sdxl", False)

    outer = self

    def process_sample(*args, **kwargs):
        # ControlNet must know whether a prompt is conditional prompt (positive prompt) or unconditional conditioning prompt (negative prompt).
        # You can use the hook.py's `mark_prompt_context` to mark the prompts that will be seen by ControlNet.
        # Let us say XXX is a MulticondLearnedConditioning or a ComposableScheduledPromptConditioning or a ScheduledPromptConditioning or a list of these components,
        # if XXX is a positive prompt, you should call mark_prompt_context(XXX, positive=True)
        # if XXX is a negative prompt, you should call mark_prompt_context(XXX, positive=False)
        # After you mark the prompts, the ControlNet will know which prompt is cond/uncond and works as expected.
        # After you mark the prompts, the mismatch errors will disappear.
        mark_prompt_context(kwargs.get("conditioning", []), positive=True)
        mark_prompt_context(
            kwargs.get("unconditional_conditioning", []), positive=False
        )
        mark_prompt_context(getattr(process, "hr_c", []), positive=True)
        mark_prompt_context(getattr(process, "hr_uc", []), positive=False)
        return process.sample_before_CN_hack(*args, **kwargs)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        is_sdxl = y is not None and model_is_sdxl
        total_t2i_adapter_embedding = [0.0] * 4
        if is_sdxl:
            total_controlnet_embedding = [0.0] * 10
        else:
            total_controlnet_embedding = [0.0] * 13
        require_inpaint_hijack = False
        is_in_high_res_fix = False
        batch_size = int(x.shape[0])

        # Handle cond-uncond marker
        (
            cond_mark,
            outer.current_uc_indices,
            outer.current_c_indices,
            context,
        ) = unmark_prompt_context(context)
        outer.model.cond_mark = cond_mark
        # logger.info(str(cond_mark[:, 0, 0, 0].detach().cpu().numpy().tolist()) + ' - ' + str(outer.current_uc_indices))

        # Revision
        if is_sdxl:
            revision_y1280 = 0

            for param in outer.control_params:
                if param.guidance_stopped:
                    continue
                if param.control_model_type == ControlModelType.ReVision:
                    if param.vision_hint_count is None:
                        k = (
                            torch.Tensor(
                                [int(param.preprocessor["threshold_a"] * 1000)]
                            )
                            .to(param.hint_cond)
                            .long()
                            .clip(0, 999)
                        )
                        param.vision_hint_count = outer.revision_q_sampler.q_sample(
                            param.hint_cond, k
                        )
                    revision_emb = param.vision_hint_count
                    if isinstance(revision_emb, torch.Tensor):
                        revision_y1280 += revision_emb * param.weight

            if isinstance(revision_y1280, torch.Tensor):
                y[:, :1280] = revision_y1280 * cond_mark[:, :, 0, 0]
                if any(
                    "ignore_prompt" in param.preprocessor["name"]
                    for param in outer.control_params
                ) or (
                    getattr(process, "prompt", "") == ""
                    and getattr(process, "negative_prompt", "") == ""
                ):
                    context = torch.zeros_like(context)

        # High-res fix
        for param in outer.control_params:
            # select which hint_cond to use
            if param.used_hint_cond is None:
                param.used_hint_cond = param.hint_cond
                param.used_hint_cond_latent = None
                param.used_hint_inpaint_hijack = None

            # has high-res fix
            if (
                isinstance(param.hr_hint_cond, torch.Tensor)
                and x.ndim == 4
                and param.hint_cond.ndim == 4
                and param.hr_hint_cond.ndim == 4
            ):
                _, _, h_lr, w_lr = param.hint_cond.shape
                _, _, h_hr, w_hr = param.hr_hint_cond.shape
                _, _, h, w = x.shape
                h, w = h * 8, w * 8
                if abs(h - h_lr) < abs(h - h_hr):
                    is_in_high_res_fix = False
                    if param.used_hint_cond is not param.hint_cond:
                        param.used_hint_cond = param.hint_cond
                        param.used_hint_cond_latent = None
                        param.used_hint_inpaint_hijack = None
                else:
                    is_in_high_res_fix = True
                    if param.used_hint_cond is not param.hr_hint_cond:
                        param.used_hint_cond = param.hr_hint_cond
                        param.used_hint_cond_latent = None
                        param.used_hint_inpaint_hijack = None

        self.is_in_high_res_fix = is_in_high_res_fix
        outer.is_in_high_res_fix = is_in_high_res_fix

        # Convert control image to latent
        for param in outer.control_params:
            if param.used_hint_cond_latent is not None:
                continue
            if (
                param.control_model_type not in [ControlModelType.AttentionInjection]
                and "colorfix" not in param.preprocessor["name"]
                and "inpaint_only" not in param.preprocessor["name"]
            ):
                continue
            param.used_hint_cond_latent = outer.call_vae_using_process(
                process, param.used_hint_cond, batch_size=batch_size
            )

        # vram
        for param in outer.control_params:
            if getattr(param.control_model, "disable_memory_management", False):
                continue

            if param.control_model is not None:
                if (
                    outer.lowvram
                    and is_sdxl
                    and hasattr(param.control_model, "aggressive_lowvram")
                ):
                    param.control_model.aggressive_lowvram()
                elif hasattr(param.control_model, "fullvram"):
                    param.control_model.fullvram()
                elif hasattr(param.control_model, "to"):
                    param.control_model.to(devices.get_device_for("controlnet"))

        # handle prompt token control
        for param in outer.control_params:
            if param.guidance_stopped or param.disabled_by_hr_option(
                self.is_in_high_res_fix
            ):
                continue

            if param.control_model_type not in [ControlModelType.T2I_StyleAdapter]:
                continue

            control = param.control_model(
                x=x, hint=param.used_hint_cond, timesteps=timesteps, context=context
            )
            control = torch.cat([control.clone() for _ in range(batch_size)], dim=0)
            control *= param.weight
            control *= cond_mark[:, :, :, 0]
            context = torch.cat([context, control.clone()], dim=1)

        # handle ControlNet / T2I_Adapter
        for param_index, param in enumerate(outer.control_params):
            if param.guidance_stopped or param.disabled_by_hr_option(
                self.is_in_high_res_fix
            ):
                continue

            if not (
                param.control_model_type.is_controlnet
                or param.control_model_type == ControlModelType.T2I_Adapter
            ):
                continue

            # inpaint model workaround
            x_in = x
            control_model = param.control_model.control_model

            if param.control_model_type.is_controlnet:
                if (
                    x.shape[1] != control_model.input_blocks[0][0].in_channels
                    and x.shape[1] == 9
                ):
                    # inpaint_model: 4 data + 4 downscaled image + 1 mask
                    x_in = x[:, :4, ...]
                    require_inpaint_hijack = True

            assert (
                param.used_hint_cond is not None
            ), "Controlnet is enabled but no input image is given"

            hint = param.used_hint_cond
            if param.control_model_type == ControlModelType.InstantID:
                assert isinstance(param.control_context_override, ImageEmbed)
                controlnet_context = param.control_context_override.eval(cond_mark).to(
                    x.device, dtype=x.dtype
                )
            else:
                controlnet_context = context

            # ControlNet inpaint protocol
            if hint.shape[1] == 4 and not isinstance(control_model, SparseCtrl):
                c = hint[:, 0:3, :, :]
                m = hint[:, 3:4, :, :]
                m = (m > 0.5).float()
                hint = c * (1 - m) - m

            control = param.control_model(
                x=x_in, hint=hint, timesteps=timesteps, context=controlnet_context, y=y
            )

            if is_sdxl:
                control_scales = [param.weight] * 10
            else:
                control_scales = [param.weight] * 13

            if param.cfg_injection or param.global_average_pooling:
                if param.control_model_type == ControlModelType.T2I_Adapter:
                    control = [
                        torch.cat([c.clone() for _ in range(batch_size)], dim=0)
                        for c in control
                    ]
                control = [c * cond_mark for c in control]

            high_res_fix_forced_soft_injection = False

            if is_in_high_res_fix:
                if "canny" in param.preprocessor["name"]:
                    high_res_fix_forced_soft_injection = True
                if "mlsd" in param.preprocessor["name"]:
                    high_res_fix_forced_soft_injection = True

            if param.soft_injection or high_res_fix_forced_soft_injection:
                # important! use the soft weights with high-res fix can significantly reduce artifacts.
                if param.control_model_type == ControlModelType.T2I_Adapter:
                    control_scales = [
                        param.weight * x for x in (0.25, 0.62, 0.825, 1.0)
                    ]
                elif param.control_model_type.is_controlnet:
                    control_scales = [
                        param.weight * (0.825 ** float(12 - i)) for i in range(13)
                    ]

            if is_sdxl and param.control_model_type.is_controlnet:
                control_scales = control_scales[:10]

            if param.advanced_weighting is not None:
                logger.info(f"Advanced weighting enabled. {param.advanced_weighting}")
                if param.soft_injection or high_res_fix_forced_soft_injection:
                    logger.warn("Advanced weighting overwrites soft_injection effect.")
                control_scales = param.advanced_weighting

            control = [
                param.apply_effective_region_mask(c * scale)
                for c, scale in zip(control, control_scales)
            ]
            if param.global_average_pooling:
                control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]

            for idx, item in enumerate(control):
                target = None
                if param.control_model_type.is_controlnet:
                    target = total_controlnet_embedding
                if param.control_model_type == ControlModelType.T2I_Adapter:
                    target = total_t2i_adapter_embedding
                if target is not None:
                    if batch_option_uint_separate:
                        for pi, ci in enumerate(outer.current_c_indices):
                            if pi % len(outer.control_params) != param_index:
                                item[ci] = 0
                        for pi, ci in enumerate(outer.current_uc_indices):
                            if pi % len(outer.control_params) != param_index:
                                item[ci] = 0
                        target[idx] = item + target[idx]
                    else:
                        target[idx] = item + target[idx]

        # Replace x_t to support inpaint models
        for param in outer.control_params:
            if not isinstance(param.used_hint_cond, torch.Tensor):
                continue
            if param.used_hint_cond.ndim < 2 or param.used_hint_cond.shape[1] != 4:
                continue
            if x.shape[1] != 9:
                continue
            if param.used_hint_inpaint_hijack is None:
                mask_pixel = param.used_hint_cond[:, 3:4, :, :]
                image_pixel = param.used_hint_cond[:, 0:3, :, :]
                mask_pixel = (mask_pixel > 0.5).to(mask_pixel.dtype)
                masked_latent = outer.call_vae_using_process(
                    process, image_pixel, batch_size, mask=mask_pixel
                )
                mask_latent = torch.nn.functional.max_pool2d(mask_pixel, (8, 8))
                if mask_latent.shape[0] != batch_size:
                    mask_latent = torch.cat(
                        [mask_latent.clone() for _ in range(batch_size)], dim=0
                    )
                param.used_hint_inpaint_hijack = torch.cat(
                    [mask_latent, masked_latent], dim=1
                )
                param.used_hint_inpaint_hijack.to(x.dtype).to(x.device)
            x = torch.cat([x[:, :4, :, :], param.used_hint_inpaint_hijack], dim=1)

        # vram
        for param in outer.control_params:
            if param.control_model is not None:
                if outer.lowvram:
                    param.control_model.to("cpu")

        # A1111 fix for medvram.
        if shared.cmd_opts.medvram or (
            getattr(shared.cmd_opts, "medvram_sdxl", False) and is_sdxl
        ):
            try:
                # Trigger the register_forward_pre_hook
                outer.sd_ldm.model()
            except Exception as e:
                logger.debug("register_forward_pre_hook")
                logger.debug(e)

        # Clear attention and AdaIn cache
        for module in outer.attn_module_list:
            module.bank = []
            module.style_cfgs = []
        for module in outer.gn_module_list:
            module.mean_bank = []
            module.var_bank = []
            module.style_cfgs = []

        # Handle attention and AdaIn control
        for param in outer.control_params:
            if param.guidance_stopped or param.disabled_by_hr_option(
                self.is_in_high_res_fix
            ):
                continue

            if param.used_hint_cond_latent is None:
                continue

            if param.control_model_type not in [ControlModelType.AttentionInjection]:
                continue

            ref_xt = predict_q_sample(
                outer.sd_ldm,
                param.used_hint_cond_latent,
                torch.round(timesteps.float()).long(),
            )

            # Inpaint Hijack
            if x.shape[1] == 9:
                ref_xt = torch.cat(
                    [
                        ref_xt,
                        torch.zeros_like(ref_xt)[:, 0:1, :, :],
                        param.used_hint_cond_latent,
                    ],
                    dim=1,
                )

            outer.current_style_fidelity = float(param.preprocessor["threshold_a"])
            outer.current_style_fidelity = max(
                0.0, min(1.0, outer.current_style_fidelity)
            )

            if is_sdxl:
                # sdxl's attention hacking is highly unstable.
                # We have no other methods but to reduce the style_fidelity a bit.
                # By default, 0.5 ** 3.0 = 0.125
                outer.current_style_fidelity = outer.current_style_fidelity**3.0

            if param.cfg_injection:
                outer.current_style_fidelity = 1.0
            elif param.soft_injection or is_in_high_res_fix:
                outer.current_style_fidelity = 0.0

            control_name = param.preprocessor["name"]

            if control_name in ["reference_only", "reference_adain+attn"]:
                outer.attention_auto_machine = AutoMachine.Write
                outer.attention_auto_machine_weight = param.weight

            if control_name in ["reference_adain", "reference_adain+attn"]:
                outer.gn_auto_machine = AutoMachine.Write
                outer.gn_auto_machine_weight = param.weight

            if is_sdxl:
                outer.original_forward(
                    x=ref_xt.to(devices.dtype_unet),
                    timesteps=timesteps.to(devices.dtype_unet),
                    context=context.to(devices.dtype_unet),
                    y=y,
                )
            else:
                outer.original_forward(
                    x=ref_xt.to(devices.dtype_unet),
                    timesteps=timesteps.to(devices.dtype_unet),
                    context=context.to(devices.dtype_unet),
                )

            outer.attention_auto_machine = AutoMachine.Read
            outer.gn_auto_machine = AutoMachine.Read

        # ------ modified by OneDiff below ------
        x = x.half()
        context = context.half()
        y = y.half() if y is not None else y
        h = onediff_shared.current_unet_graph.graph_module(
            x,
            timesteps,
            context,
            y,
            total_t2i_adapter_embedding,
            total_controlnet_embedding,
            is_sdxl,
            require_inpaint_hijack,
        )
        # ------ modified by OneDiff above ------

        # Post-processing for color fix
        for param in outer.control_params:
            if param.used_hint_cond_latent is None:
                continue
            if "colorfix" not in param.preprocessor["name"]:
                continue

            k = int(param.preprocessor["threshold_a"])
            if is_in_high_res_fix and not param.disabled_by_hr_option(
                self.is_in_high_res_fix
            ):
                k *= 2

            # Inpaint hijack
            xt = x[:, :4, :, :]

            x0_origin = param.used_hint_cond_latent
            t = torch.round(timesteps.float()).long()
            x0_prd = predict_start_from_noise(outer.sd_ldm, xt, t, h)
            x0 = x0_prd - blur(x0_prd, k) + blur(x0_origin, k)

            if "+sharp" in param.preprocessor["name"]:
                detail_weight = float(param.preprocessor["threshold_b"]) * 0.01
                neg = detail_weight * blur(x0, k) + (1 - detail_weight) * x0
                x0 = cond_mark * x0 + (1 - cond_mark) * neg

            eps_prd = predict_noise_from_start(outer.sd_ldm, xt, t, x0)

            w = max(0.0, min(1.0, float(param.weight)))
            h = eps_prd * w + h * (1 - w)

        # Post-processing for restore
        for param in outer.control_params:
            if param.used_hint_cond_latent is None:
                continue
            if "inpaint_only" not in param.preprocessor["name"]:
                continue
            if param.used_hint_cond.shape[1] != 4:
                continue

            # Inpaint hijack
            xt = x[:, :4, :, :]

            mask = param.used_hint_cond[:, 3:4, :, :]
            mask = torch.nn.functional.max_pool2d(
                mask, (10, 10), stride=(8, 8), padding=1
            )

            x0_origin = param.used_hint_cond_latent
            t = torch.round(timesteps.float()).long()
            x0_prd = predict_start_from_noise(outer.sd_ldm, xt, t, h)
            x0 = x0_prd * mask + x0_origin * (1 - mask)
            eps_prd = predict_noise_from_start(outer.sd_ldm, xt, t, x0)

            w = max(0.0, min(1.0, float(param.weight)))
            h = eps_prd * w + h * (1 - w)

        return h

    def move_all_control_model_to_cpu():
        for param in getattr(outer, "control_params", []) or []:
            if isinstance(param.control_model, torch.nn.Module):
                param.control_model.to("cpu")

    def forward_webui(*args, **kwargs):
        # ------ modified by OneDiff below ------
        forward_func = None
        graph_module = onediff_shared.current_unet_graph.graph_module
        if is_oneflow_backend():
            if "forward" in graph_module._torch_module.__dict__:
                forward_func = graph_module._torch_module.__dict__.pop("forward")
                _original_forward_func = graph_module._torch_module.__dict__.pop(
                    "_original_forward"
                )
        # ------ modified by OneDiff above ------

        # webui will handle other compoments
        try:
            if shared.cmd_opts.lowvram:
                lowvram.send_everything_to_cpu()
            return forward(*args, **kwargs)
        except Exception as e:
            move_all_control_model_to_cpu()
            raise e
        finally:
            if outer.lowvram:
                move_all_control_model_to_cpu()

            # ------ modified by OneDiff below ------
            if is_oneflow_backend():
                if forward_func is not None:
                    graph_module._torch_module.forward = forward_func
                    graph_module._torch_module._original_forward = (
                        _original_forward_func
                    )
            # ------ modified by OneDiff above ------

    def hacked_basic_transformer_inner_forward(self, x, context=None):
        x_norm1 = self.norm1(x)
        self_attn1 = None
        if self.disable_self_attn:
            # Do not use self-attention
            self_attn1 = self.attn1(x_norm1, context=context)
        else:
            # Use self-attention
            self_attention_context = x_norm1
            if outer.attention_auto_machine == AutoMachine.Write:
                if outer.attention_auto_machine_weight > self.attn_weight:
                    self.bank.append(self_attention_context.detach().clone())
                    self.style_cfgs.append(outer.current_style_fidelity)
            if outer.attention_auto_machine == AutoMachine.Read:
                if len(self.bank) > 0:
                    style_cfg = sum(self.style_cfgs) / float(len(self.style_cfgs))
                    self_attn1_uc = self.attn1(
                        x_norm1,
                        context=torch.cat([self_attention_context] + self.bank, dim=1),
                    )
                    self_attn1_c = self_attn1_uc.clone()
                    if len(outer.current_uc_indices) > 0 and style_cfg > 1e-5:
                        self_attn1_c[outer.current_uc_indices] = self.attn1(
                            x_norm1[outer.current_uc_indices],
                            context=self_attention_context[outer.current_uc_indices],
                        )
                    self_attn1 = (
                        style_cfg * self_attn1_c + (1.0 - style_cfg) * self_attn1_uc
                    )
                self.bank = []
                self.style_cfgs = []
            if (
                outer.attention_auto_machine == AutoMachine.StyleAlign
                and not outer.is_in_high_res_fix
            ):
                # very VRAM hungry - disable at high_res_fix

                def shared_attn1(inner_x):
                    BB, FF, CC = inner_x.shape
                    return self.attn1(inner_x.reshape(1, BB * FF, CC)).reshape(
                        BB, FF, CC
                    )

                uc_layer = shared_attn1(x_norm1[outer.current_uc_indices])
                c_layer = shared_attn1(x_norm1[outer.current_c_indices])
                self_attn1 = torch.zeros_like(x_norm1).to(uc_layer)
                self_attn1[outer.current_uc_indices] = uc_layer
                self_attn1[outer.current_c_indices] = c_layer
                del uc_layer, c_layer
            if self_attn1 is None:
                self_attn1 = self.attn1(x_norm1, context=self_attention_context)

        x = self_attn1.to(x.dtype) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

    def hacked_group_norm_forward(self, *args, **kwargs):
        eps = 1e-6
        x = self.original_forward_cn_hijack(*args, **kwargs)
        y = None
        if outer.gn_auto_machine == AutoMachine.Write:
            if outer.gn_auto_machine_weight > self.gn_weight:
                var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                self.mean_bank.append(mean)
                self.var_bank.append(var)
                self.style_cfgs.append(outer.current_style_fidelity)
        if outer.gn_auto_machine == AutoMachine.Read:
            if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                style_cfg = sum(self.style_cfgs) / float(len(self.style_cfgs))
                var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                mean_acc = sum(self.mean_bank) / float(len(self.mean_bank))
                var_acc = sum(self.var_bank) / float(len(self.var_bank))
                std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                y_uc = (((x - mean) / std) * std_acc) + mean_acc
                y_c = y_uc.clone()
                if len(outer.current_uc_indices) > 0 and style_cfg > 1e-5:
                    y_c[outer.current_uc_indices] = x.to(y_c.dtype)[
                        outer.current_uc_indices
                    ]
                y = style_cfg * y_c + (1.0 - style_cfg) * y_uc
            self.mean_bank = []
            self.var_bank = []
            self.style_cfgs = []
        if y is None:
            y = x
        return y.to(x.dtype)

    if getattr(process, "sample_before_CN_hack", None) is None:
        process.sample_before_CN_hack = process.sample
    process.sample = process_sample

    model._original_forward = model.forward
    outer.original_forward = model.forward
    model.forward = forward_webui.__get__(model, UNetModel)

    if model_is_sdxl:
        register_schedule(sd_ldm)
        outer.revision_q_sampler = AbstractLowScaleModel()

    need_attention_hijack = False

    for param in outer.control_params:
        if param.control_model_type in [ControlModelType.AttentionInjection]:
            need_attention_hijack = True

    if batch_option_style_align:
        need_attention_hijack = True
        outer.attention_auto_machine = AutoMachine.StyleAlign
        outer.gn_auto_machine = AutoMachine.StyleAlign

    all_modules = torch_dfs(model)

    if need_attention_hijack:
        attn_modules = [
            module
            for module in all_modules
            if isinstance(module, BasicTransformerBlock)
            or isinstance(module, BasicTransformerBlockSGM)
        ]
        attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

        for i, module in enumerate(attn_modules):
            if getattr(module, "_original_inner_forward_cn_hijack", None) is None:
                module._original_inner_forward_cn_hijack = module._forward
            module._forward = hacked_basic_transformer_inner_forward.__get__(
                module, BasicTransformerBlock
            )
            module.bank = []
            module.style_cfgs = []
            module.attn_weight = float(i) / float(len(attn_modules))

        gn_modules = [model.middle_block]
        model.middle_block.gn_weight = 0

        if model_is_sdxl:
            input_block_indices = [4, 5, 7, 8]
            output_block_indices = [0, 1, 2, 3, 4, 5]
        else:
            input_block_indices = [4, 5, 7, 8, 10, 11]
            output_block_indices = [0, 1, 2, 3, 4, 5, 6, 7]

        for w, i in enumerate(input_block_indices):
            module = model.input_blocks[i]
            module.gn_weight = 1.0 - float(w) / float(len(input_block_indices))
            gn_modules.append(module)

        for w, i in enumerate(output_block_indices):
            module = model.output_blocks[i]
            module.gn_weight = float(w) / float(len(output_block_indices))
            gn_modules.append(module)

        for i, module in enumerate(gn_modules):
            if getattr(module, "original_forward_cn_hijack", None) is None:
                module.original_forward_cn_hijack = module.forward
            module.forward = hacked_group_norm_forward.__get__(module, torch.nn.Module)
            module.mean_bank = []
            module.var_bank = []
            module.style_cfgs = []
            module.gn_weight *= 2

        outer.attn_module_list = attn_modules
        outer.gn_module_list = gn_modules
    else:
        for module in all_modules:
            _original_inner_forward_cn_hijack = getattr(
                module, "_original_inner_forward_cn_hijack", None
            )
            original_forward_cn_hijack = getattr(
                module, "original_forward_cn_hijack", None
            )
            if _original_inner_forward_cn_hijack is not None:
                module._forward = _original_inner_forward_cn_hijack
            if original_forward_cn_hijack is not None:
                module.forward = original_forward_cn_hijack
        outer.attn_module_list = []
        outer.gn_module_list = []

    scripts.script_callbacks.on_cfg_denoiser(self.guidance_schedule_handler)
