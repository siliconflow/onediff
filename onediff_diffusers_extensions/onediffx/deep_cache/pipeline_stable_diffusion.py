# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Dict, List, Optional, Union

from packaging import version
import importlib.metadata

diffusers_0202_v = version.parse("0.20.2")
diffusers_0214_v = version.parse("0.21.4")
diffusers_0223_v = version.parse("0.22.3")
diffusers_0231_v = version.parse("0.23.1")
diffusers_0240_v = version.parse("0.24.0")
diffusers_0251_v = version.parse("0.25.1")
diffusers_0263_v = version.parse("0.26.3")
diffusers_0270_v = version.parse("0.27.0")
diffusers_version = version.parse(importlib.metadata.version("diffusers"))

import torch
import numpy as np
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

if diffusers_version >= diffusers_0240_v:
    from transformers import CLIPVisionModelWithProjection
    from diffusers.image_processor import PipelineImageInput
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    logging,
)

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

from diffusers import StableDiffusionPipeline as DiffusersStableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg

from .models.unet_2d_condition import UNet2DConditionModel
from .models.fast_unet_2d_condition import FastUNet2DConditionModel


from .models.pipeline_utils import enable_deep_cache_pipeline


enable_deep_cache_pipeline()

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def sample_from_quad(total_numbers, n_samples, pow=1.2):
    while pow > 1:
        # Generate linearly spaced values between 0 and a max value
        x_values = np.linspace(0, total_numbers ** (1 / pow), n_samples + 1)

        # Raise these values to the power of 1.5 to get a non-linear distribution
        indices = np.unique(np.int32(x_values ** pow))[:-1]
        if len(indices) == n_samples:
            break
        pow -= 0.02
    if pow <= 1:
        raise ValueError(
            "Cannot find suitable pow. Please adjust n_samples or decrease center."
        )
    return indices, pow


def sample_from_quad_center(total_numbers, n_samples, center, pow=1.2):
    while pow > 1:
        # Generate linearly spaced values between 0 and a max value
        x_values = np.linspace(
            (-center) ** (1 / pow), (total_numbers - center) ** (1 / pow), n_samples + 1
        )
        indices = [0] + [x + center for x in np.unique(np.int32(x_values ** pow))[1:-1]]
        if len(indices) == n_samples:
            break
        pow -= 0.02
    if pow <= 1:
        raise ValueError(
            "Cannot find suitable pow. Please adjust n_samples or decrease center."
        )
    return indices, pow

if diffusers_version <= diffusers_0214_v:
    class StableDiffusionPipeline(DiffusersStableDiffusionPipeline):
        _optional_components = ["safety_checker", "feature_extractor"]
        if diffusers_version > diffusers_0202_v:
            model_cpu_offload_seq = "text_encoder->unet->vae"
            _exclude_from_cpu_offload = ["safety_checker"]

        def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            requires_safety_checker: bool = True,
        ):
            if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
                deprecation_message = (
                    f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                    f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                    "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                    " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                    " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                    " file"
                )
                deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
                new_config = dict(scheduler.config)
                new_config["steps_offset"] = 1
                scheduler._internal_dict = FrozenDict(new_config)

            if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
                deprecation_message = (
                    f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                    " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                    " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                    " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                    " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
                )
                deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
                new_config = dict(scheduler.config)
                new_config["clip_sample"] = False
                scheduler._internal_dict = FrozenDict(new_config)

            if safety_checker is None and requires_safety_checker:
                logger.warning(
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )

            if safety_checker is not None and feature_extractor is None:
                raise ValueError(
                    "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                    " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
                )

            is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
                version.parse(unet.config._diffusers_version).base_version
            ) < version.parse("0.9.0.dev0")
            is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
            if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
                deprecation_message = (
                    "The configuration file of the unet has set the default `sample_size` to smaller than"
                    " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                    " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                    " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                    " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                    " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                    " in the config might lead to incorrect results in future versions. If you have downloaded this"
                    " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                    " the `unet/config.json` file"
                )
                deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
                new_config = dict(unet.config)
                new_config["sample_size"] = 64
                unet._internal_dict = FrozenDict(new_config)

            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
            )
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            self.register_to_config(requires_safety_checker=requires_safety_checker)
            self.fast_unet = FastUNet2DConditionModel(self.unet)
        
        @torch.no_grad()
        def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            cache_interval: int = 1,
            cache_layer_id: int = None,
            cache_block_id: int = None,
            uniform: bool = True,
            pow: float = None,
            center: int = None,
        ):
            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
            )

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            # 3. Encode input prompt
            text_encoder_lora_scale = (
                cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
            )
            if diffusers_version > diffusers_0202_v:
                prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                    prompt,
                    device,
                    num_images_per_prompt,
                    do_classifier_free_guidance,
                    negative_prompt,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    lora_scale=text_encoder_lora_scale,
                )
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                if do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            else:
                prompt_embeds = self._encode_prompt(
                    prompt,
                    device,
                    num_images_per_prompt,
                    do_classifier_free_guidance,
                    negative_prompt,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    lora_scale=text_encoder_lora_scale,
                )

            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 5. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

            prv_features = None
            latents_list = [latents]

            if cache_interval == 1:
                interval_seq = list(range(num_inference_steps))
            else:
                if uniform:
                    interval_seq = list(range(0, num_inference_steps, cache_interval))
                else:
                    num_slow_step = num_inference_steps // cache_interval
                    if num_inference_steps % cache_interval != 0:
                        num_slow_step += 1

                    interval_seq, pow = sample_from_quad_center(
                        num_inference_steps, num_slow_step, center=center, pow=pow
                    )  # [0, 3, 6, 9, 12, 16, 22, 28, 35, 43,]
                    # interval_seq, pow = sample_from_quad(num_inference_steps, num_inference_steps//cache_interval, pow=pow)#[0, 3, 6, 9, 12, 16, 22, 28, 35, 43,]

            interval_seq = sorted(interval_seq)
            
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    if i in interval_seq or cache_interval == 1:
                        prv_features = None
                        # print(t, prv_features is None)
                        # predict the noise residual

                        noise_pred, prv_features = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            replicate_prv_feature=prv_features,
                            cache_layer_id=cache_layer_id,
                            cache_block_id=cache_block_id,
                            return_dict=False,
                        )
                    else:
                        noise_pred, prv_features = self.fast_unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            replicate_prv_feature=prv_features,
                            cache_layer_id=cache_layer_id,
                            cache_block_id=cache_block_id,
                            return_dict=False,
                        )
                    
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)
            if not output_type == "latent":
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            else:
                image = latents
                has_nsfw_concept = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

            if diffusers_version > diffusers_0202_v:
                # Offload all models
                self.maybe_free_model_hooks()
            else:
                # Offload last model to CPU
                if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                    self.final_offload_hook.offload()

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

elif diffusers_version <= diffusers_0231_v:
    class StableDiffusionPipeline(DiffusersStableDiffusionPipeline):
        _optional_components = ["safety_checker", "feature_extractor"]
        model_cpu_offload_seq = "text_encoder->unet->vae"
        _exclude_from_cpu_offload = ["safety_checker"]
        _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

        def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            requires_safety_checker: bool = True,
        ):
            if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
                deprecation_message = (
                    f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                    f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                    "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                    " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                    " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                    " file"
                )
                deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
                new_config = dict(scheduler.config)
                new_config["steps_offset"] = 1
                scheduler._internal_dict = FrozenDict(new_config)

            if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
                deprecation_message = (
                    f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                    " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                    " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                    " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                    " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
                )
                deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
                new_config = dict(scheduler.config)
                new_config["clip_sample"] = False
                scheduler._internal_dict = FrozenDict(new_config)

            if safety_checker is None and requires_safety_checker:
                logger.warning(
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )

            if safety_checker is not None and feature_extractor is None:
                raise ValueError(
                    "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                    " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
                )

            is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
                version.parse(unet.config._diffusers_version).base_version
            ) < version.parse("0.9.0.dev0")
            is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
            if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
                deprecation_message = (
                    "The configuration file of the unet has set the default `sample_size` to smaller than"
                    " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                    " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                    " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                    " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                    " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                    " in the config might lead to incorrect results in future versions. If you have downloaded this"
                    " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                    " the `unet/config.json` file"
                )
                deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
                new_config = dict(unet.config)
                new_config["sample_size"] = 64
                unet._internal_dict = FrozenDict(new_config)

            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
            )
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            self.register_to_config(requires_safety_checker=requires_safety_checker)
            self.fast_unet = FastUNet2DConditionModel(self.unet)
        
        @torch.no_grad()
        def __call__(
            self,
             prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            cache_interval: int = 1,
            cache_layer_id: int = None,
            cache_block_id: int = None,
            uniform: bool = True,
            pow: float = None,
            center: int = None,
            **kwargs,
        ):
            callback = kwargs.pop("callback", None)
            callback_steps = kwargs.pop("callback_steps", None)

            if callback is not None:
                deprecate(
                    "callback",
                    "1.0.0",
                    "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )
            if callback_steps is not None:
                deprecate(
                    "callback_steps",
                    "1.0.0",
                    "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )
            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor
            # to deal with lora scaling and other possible forward hooks

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                height,
                width,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                callback_on_step_end_tensor_inputs,
            )

            self._guidance_scale = guidance_scale
            self._guidance_rescale = guidance_rescale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = cross_attention_kwargs

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            # 3. Encode input prompt
            lora_scale = (
                self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
            )

            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 5. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            if diffusers_version > diffusers_0223_v:
                # 6.5 Optionally get Guidance Scale Embedding
                timestep_cond = None
                if self.unet.config.time_cond_proj_dim is not None:
                    guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                    timestep_cond = self.get_guidance_scale_embedding(
                        guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                    ).to(device=device, dtype=latents.dtype)

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)

            prv_features = None
            latents_list = [latents]

            if cache_interval == 1:
                interval_seq = list(range(num_inference_steps))
            else:
                if uniform:
                    interval_seq = list(range(0, num_inference_steps, cache_interval))
                else:
                    num_slow_step = num_inference_steps // cache_interval
                    if num_inference_steps % cache_interval != 0:
                        num_slow_step += 1

                    interval_seq, pow = sample_from_quad_center(
                        num_inference_steps, num_slow_step, center=center, pow=pow
                    )  # [0, 3, 6, 9, 12, 16, 22, 28, 35, 43,]
                    # interval_seq, pow = sample_from_quad(num_inference_steps, num_inference_steps//cache_interval, pow=pow)#[0, 3, 6, 9, 12, 16, 22, 28, 35, 43,]

            interval_seq = sorted(interval_seq)
            
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    if i in interval_seq or cache_interval == 1:
                        prv_features = None
                        # print(t, prv_features is None)
                        # predict the noise residual

                        if diffusers_version > diffusers_0223_v:
                            noise_pred, prv_features = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                timestep_cond=timestep_cond,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                replicate_prv_feature=prv_features,
                                cache_layer_id=cache_layer_id,
                                cache_block_id=cache_block_id,
                                return_dict=False,
                            )
                        else:
                            noise_pred, prv_features = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                replicate_prv_feature=prv_features,
                                cache_layer_id=cache_layer_id,
                                cache_block_id=cache_block_id,
                                return_dict=False,
                            )
                    else:
                        if diffusers_version > diffusers_0223_v:
                            noise_pred, prv_features = self.fast_unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                timestep_cond=timestep_cond,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                replicate_prv_feature=prv_features,
                                cache_layer_id=cache_layer_id,
                                cache_block_id=cache_block_id,
                                return_dict=False,
                            )
                        else:
                            noise_pred, prv_features = self.fast_unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                replicate_prv_feature=prv_features,
                                cache_layer_id=cache_layer_id,
                                cache_block_id=cache_block_id,
                                return_dict=False,
                            )
                    
                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)
            if not output_type == "latent":
                if diffusers_version > diffusers_0223_v:
                    image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                        0
                    ]
                else:
                    image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            else:
                image = latents
                has_nsfw_concept = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

            if diffusers_version > diffusers_0202_v:
                # Offload all models
                self.maybe_free_model_hooks()
            else:
                # Offload last model to CPU
                if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                    self.final_offload_hook.offload()

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

elif diffusers_version < diffusers_0270_v:
    class StableDiffusionPipeline(DiffusersStableDiffusionPipeline):
        if diffusers_version > diffusers_0240_v:
            model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
        else:
            model_cpu_offload_seq = "text_encoder->unet->vae"
        _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
        _exclude_from_cpu_offload = ["safety_checker"]
        _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

        def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            image_encoder: CLIPVisionModelWithProjection = None,
            requires_safety_checker: bool = True,
        ):
            if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
                deprecation_message = (
                    f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                    f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                    "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                    " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                    " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                    " file"
                )
                deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
                new_config = dict(scheduler.config)
                new_config["steps_offset"] = 1
                scheduler._internal_dict = FrozenDict(new_config)

            if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
                deprecation_message = (
                    f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                    " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                    " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                    " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                    " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
                )
                deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
                new_config = dict(scheduler.config)
                new_config["clip_sample"] = False
                scheduler._internal_dict = FrozenDict(new_config)

            if safety_checker is None and requires_safety_checker:
                logger.warning(
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )

            if safety_checker is not None and feature_extractor is None:
                raise ValueError(
                    "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                    " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
                )

            is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
                version.parse(unet.config._diffusers_version).base_version
            ) < version.parse("0.9.0.dev0")
            is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
            if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
                deprecation_message = (
                    "The configuration file of the unet has set the default `sample_size` to smaller than"
                    " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                    " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                    " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                    " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                    " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                    " in the config might lead to incorrect results in future versions. If you have downloaded this"
                    " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                    " the `unet/config.json` file"
                )
                deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
                new_config = dict(unet.config)
                new_config["sample_size"] = 64
                unet._internal_dict = FrozenDict(new_config)

            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
                image_encoder=image_encoder,
            )
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            self.register_to_config(requires_safety_checker=requires_safety_checker)
            self.fast_unet = FastUNet2DConditionModel(self.unet)
        
        @torch.no_grad()
        def __call__(
            self,
             prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            cache_interval: int = 1,
            cache_layer_id: int = None,
            cache_block_id: int = None,
            uniform: bool = True,
            pow: float = None,
            center: int = None,
            **kwargs,
        ):
            callback = kwargs.pop("callback", None)
            callback_steps = kwargs.pop("callback_steps", None)

            if callback is not None:
                deprecate(
                    "callback",
                    "1.0.0",
                    "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )
            if callback_steps is not None:
                deprecate(
                    "callback_steps",
                    "1.0.0",
                    "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )
            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor
            # to deal with lora scaling and other possible forward hooks

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                height,
                width,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                callback_on_step_end_tensor_inputs,
            )

            self._guidance_scale = guidance_scale
            self._guidance_rescale = guidance_rescale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = cross_attention_kwargs
            if diffusers_version > diffusers_0240_v:
                self._interrupt = False

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            # 3. Encode input prompt
            lora_scale = (
                self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
            )

            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            if ip_adapter_image is not None:
                if diffusers_version > diffusers_0251_v:
                    image_embeds = self.prepare_ip_adapter_image_embeds(
                        ip_adapter_image, device, batch_size * num_images_per_prompt
                    )
                else:
                    if diffusers_version > diffusers_0240_v:
                        output_hidden_state = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True
                        image_embeds, negative_image_embeds = self.encode_image(
                            ip_adapter_image, device, num_images_per_prompt, output_hidden_state
                        )
                    else:
                        image_embeds, negative_image_embeds = self.encode_image(ip_adapter_image, device, num_images_per_prompt)
                    if self.do_classifier_free_guidance:
                        image_embeds = torch.cat([negative_image_embeds, image_embeds])

            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

            # 5. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)


            # 6.1 Add image embeds for IP-Adapter
            added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

            # 6.2 Optionally get Guidance Scale Embedding
            timestep_cond = None
            if self.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                ).to(device=device, dtype=latents.dtype)

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)

            prv_features = None
            latents_list = [latents]

            if cache_interval == 1:
                interval_seq = list(range(num_inference_steps))
            else:
                if uniform:
                    interval_seq = list(range(0, num_inference_steps, cache_interval))
                else:
                    num_slow_step = num_inference_steps // cache_interval
                    if num_inference_steps % cache_interval != 0:
                        num_slow_step += 1

                    interval_seq, pow = sample_from_quad_center(
                        num_inference_steps, num_slow_step, center=center, pow=pow
                    )  # [0, 3, 6, 9, 12, 16, 22, 28, 35, 43,]
                    # interval_seq, pow = sample_from_quad(num_inference_steps, num_inference_steps//cache_interval, pow=pow)#[0, 3, 6, 9, 12, 16, 22, 28, 35, 43,]

            interval_seq = sorted(interval_seq)
            
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if diffusers_version > diffusers_0240_v:
                        if self.interrupt:
                            continue
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    if i in interval_seq or cache_interval == 1:
                        prv_features = None
                        # print(t, prv_features is None)
                        # predict the noise residual

                        if diffusers_version >= diffusers_0240_v:
                            noise_pred, prv_features = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                timestep_cond=timestep_cond,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                added_cond_kwargs=added_cond_kwargs,
                                replicate_prv_feature=prv_features,
                                cache_layer_id=cache_layer_id,
                                cache_block_id=cache_block_id,
                                return_dict=False,
                            )
                        else:
                            noise_pred, prv_features = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                timestep_cond=timestep_cond,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                replicate_prv_feature=prv_features,
                                cache_layer_id=cache_layer_id,
                                cache_block_id=cache_block_id,
                                return_dict=False,
                            )
                    else:
                        if diffusers_version >= diffusers_0240_v:
                            noise_pred, prv_features = self.fast_unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                timestep_cond=timestep_cond,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                added_cond_kwargs=added_cond_kwargs,
                                replicate_prv_feature=prv_features,
                                cache_layer_id=cache_layer_id,
                                cache_block_id=cache_block_id,
                                return_dict=False,
                            )
                        else:
                            noise_pred, prv_features = self.fast_unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                timestep_cond=timestep_cond,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                replicate_prv_feature=prv_features,
                                cache_layer_id=cache_layer_id,
                                cache_block_id=cache_block_id,
                                return_dict=False,
                            )
                    
                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)
            if not output_type == "latent":
                if diffusers_version > diffusers_0223_v:
                    image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                        0
                    ]
                else:
                    image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            else:
                image = latents
                has_nsfw_concept = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

            if diffusers_version > diffusers_0202_v:
                # Offload all models
                self.maybe_free_model_hooks()
            else:
                # Offload last model to CPU
                if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                    self.final_offload_hook.offload()

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
else:

    class StableDiffusionPipeline(DiffusersStableDiffusionPipeline):
        model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
        _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
        _exclude_from_cpu_offload = ["safety_checker"]
        _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

        def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            image_encoder: CLIPVisionModelWithProjection = None,
            requires_safety_checker: bool = True,
        ):
            if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
                deprecation_message = (
                    f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                    f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                    "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                    " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                    " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                    " file"
                )
                deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
                new_config = dict(scheduler.config)
                new_config["steps_offset"] = 1
                scheduler._internal_dict = FrozenDict(new_config)

            if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
                deprecation_message = (
                    f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                    " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                    " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                    " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                    " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
                )
                deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
                new_config = dict(scheduler.config)
                new_config["clip_sample"] = False
                scheduler._internal_dict = FrozenDict(new_config)

            if safety_checker is None and requires_safety_checker:
                logger.warning(
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )

            if safety_checker is not None and feature_extractor is None:
                raise ValueError(
                    "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                    " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
                )

            is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
                version.parse(unet.config._diffusers_version).base_version
            ) < version.parse("0.9.0.dev0")
            is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
            if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
                deprecation_message = (
                    "The configuration file of the unet has set the default `sample_size` to smaller than"
                    " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                    " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                    " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                    " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                    " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                    " in the config might lead to incorrect results in future versions. If you have downloaded this"
                    " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                    " the `unet/config.json` file"
                )
                deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
                new_config = dict(unet.config)
                new_config["sample_size"] = 64
                unet._internal_dict = FrozenDict(new_config)

            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
                image_encoder=image_encoder,
            )
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            self.register_to_config(requires_safety_checker=requires_safety_checker)
            self.fast_unet = FastUNet2DConditionModel(self.unet)

        @torch.no_grad()
        def __call__(
            self,
             prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            cache_interval: int = 1,
            cache_layer_id: int = None,
            cache_block_id: int = None,
            uniform: bool = True,
            pow: float = None,
            center: int = None,
            **kwargs,
        ):
            callback = kwargs.pop("callback", None)
            callback_steps = kwargs.pop("callback_steps", None)

            if callback is not None:
                deprecate(
                    "callback",
                    "1.0.0",
                    "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )
            if callback_steps is not None:
                deprecate(
                    "callback_steps",
                    "1.0.0",
                    "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )
            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor
            # to deal with lora scaling and other possible forward hooks

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                height,
                width,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                ip_adapter_image,
                ip_adapter_image_embeds,
                callback_on_step_end_tensor_inputs,
            )

            self._guidance_scale = guidance_scale
            self._guidance_rescale = guidance_rescale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = cross_attention_kwargs
            self._interrupt = False

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            # 3. Encode input prompt
            lora_scale = (
                self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
            )

            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                image_embeds = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                    self.do_classifier_free_guidance,
                )

            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

            # 5. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)


            # 6.1 Add image embeds for IP-Adapter
            added_cond_kwargs = (
                {"image_embeds": image_embeds}
                if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
                else None
            )

            # 6.2 Optionally get Guidance Scale Embedding
            timestep_cond = None
            if self.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                ).to(device=device, dtype=latents.dtype)

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)

            prv_features = None
            latents_list = [latents]

            if cache_interval == 1:
                interval_seq = list(range(num_inference_steps))
            else:
                if uniform:
                    interval_seq = list(range(0, num_inference_steps, cache_interval))
                else:
                    num_slow_step = num_inference_steps // cache_interval
                    if num_inference_steps % cache_interval != 0:
                        num_slow_step += 1

                    interval_seq, pow = sample_from_quad_center(
                        num_inference_steps, num_slow_step, center=center, pow=pow
                    )  # [0, 3, 6, 9, 12, 16, 22, 28, 35, 43,]
                    # interval_seq, pow = sample_from_quad(num_inference_steps, num_inference_steps//cache_interval, pow=pow)#[0, 3, 6, 9, 12, 16, 22, 28, 35, 43,]

            interval_seq = sorted(interval_seq)
            
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    if i in interval_seq or cache_interval == 1:
                        prv_features = None
                        # print(t, prv_features is None)
                        # predict the noise residual

                        noise_pred, prv_features = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            replicate_prv_feature=prv_features,
                            cache_layer_id=cache_layer_id,
                            cache_block_id=cache_block_id,
                            return_dict=False,
                        )
                    else:
                        noise_pred, prv_features = self.fast_unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            replicate_prv_feature=prv_features,
                            cache_layer_id=cache_layer_id,
                            cache_block_id=cache_block_id,
                            return_dict=False,
                        )
                    
                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)
            if not output_type == "latent":
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                    0
                ]
                image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            else:
                image = latents
                has_nsfw_concept = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
