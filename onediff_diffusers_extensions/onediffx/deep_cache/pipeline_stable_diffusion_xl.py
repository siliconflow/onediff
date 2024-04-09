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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import importlib.metadata
from packaging import version

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
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    is_invisible_watermark_available,
    logging,
)
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput

from diffusers import StableDiffusionXLPipeline as DiffusersStableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg

from .models.unet_2d_condition import UNet2DConditionModel
from .models.fast_unet_2d_condition import FastUNet2DConditionModel

if diffusers_version > diffusers_0214_v:
    from diffusers.utils import is_torch_xla_available

    if is_torch_xla_available():
        import torch_xla.core.xla_model as xm

        XLA_AVAILABLE = True
    else:
        XLA_AVAILABLE = False

if diffusers_version >= diffusers_0240_v:
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from diffusers.image_processor import PipelineImageInput
    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps

from .models.pipeline_utils import enable_deep_cache_pipeline


enable_deep_cache_pipeline()

if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import (
        StableDiffusionXLWatermarker,
    )


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

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


if diffusers_version <= diffusers_0202_v:
    class StableDiffusionXLPipeline(DiffusersStableDiffusionXLPipeline):
        def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            text_encoder_2: CLIPTextModelWithProjection,
            tokenizer: CLIPTokenizer,
            tokenizer_2: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            force_zeros_for_empty_prompt: bool = True,
            add_watermarker: Optional[bool] = None,
        ):
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                unet=unet,
                scheduler=scheduler,
            )
            self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            self.default_sample_size = self.unet.config.sample_size

            add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

            self.fast_unet = FastUNet2DConditionModel(self.unet)

            self.vae_upcasted = False

            # make sure the VAE is in float32 mode, as it overflows in float16
            self.needs_upcasting = (
                self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            )

            if add_watermarker:
                self.watermark = StableDiffusionXLWatermarker()
            else:
                self.watermark = None
        
        def upcast_vae(self):
            super().upcast_vae()
            self.vae_upcasted = True
        
        @torch.no_grad()
        def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            cache_interval: int = 1,
            cache_layer_id: int = None,
            cache_block_id: int = None,
            uniform: bool = True,
            pow: float = None,
            center: int = None,
        ):
            # 0. Default height and width to unet
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor

            original_size = original_size or (height, width)
            target_size = target_size or (height, width)

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                prompt_2,
                height,
                width,
                callback_steps,
                negative_prompt,
                negative_prompt_2,
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
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
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
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

            # 7. Prepare added time ids & embeddings
            add_text_embeds = pooled_prompt_embeds
            add_time_ids = self._get_add_time_ids(
                original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
            )

            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

            prompt_embeds = prompt_embeds.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

            # 8. Denoising loop
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

            # 7.1 Apply denoising_end
            if denoising_end is not None and type(denoising_end) == float and denoising_end > 0 and denoising_end < 1:
                discrete_timestep_cutoff = int(
                    round(
                        self.scheduler.config.num_train_timesteps
                        - (denoising_end * self.scheduler.config.num_train_timesteps)
                    )
                )
                num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
                timesteps = timesteps[:num_inference_steps]
            
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
                    )
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                    if i in interval_seq or cache_interval == 1:
                        prv_features = None
                        # print(t, prv_features is None)
                        # predict the noise residual
                        noise_pred, prv_features = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
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
                            cross_attention_kwargs=cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
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

            if self.needs_upcasting:
                if not self.vae_upcasted:
                    self.upcast_vae()
                dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype
                latents = latents.to(dtype)

            if not output_type == "latent":
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            else:
                image = latents
                return StableDiffusionXLPipelineOutput(images=image)

            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

            # Offload last model to CPU
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                self.final_offload_hook.offload()

            if not return_dict:
                return (image,)

            return StableDiffusionXLPipelineOutput(images=image)

elif diffusers_version <= diffusers_0214_v:
    class StableDiffusionXLPipeline(DiffusersStableDiffusionXLPipeline):
        model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"
        def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            text_encoder_2: CLIPTextModelWithProjection,
            tokenizer: CLIPTokenizer,
            tokenizer_2: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            force_zeros_for_empty_prompt: bool = True,
            add_watermarker: Optional[bool] = None,
        ):
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                unet=unet,
                scheduler=scheduler,
            )
            self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            self.default_sample_size = self.unet.config.sample_size

            add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

            self.fast_unet = FastUNet2DConditionModel(self.unet)

            self.vae_upcasted = False

            # make sure the VAE is in float32 mode, as it overflows in float16
            self.needs_upcasting = (
                self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            )

            if add_watermarker:
                self.watermark = StableDiffusionXLWatermarker()
            else:
                self.watermark = None
        
        def upcast_vae(self):
            super().upcast_vae()
            self.vae_upcasted = True
        
        @torch.no_grad()
        def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
            cache_interval: int = 1,
            cache_layer_id: int = None,
            cache_block_id: int = None,
            uniform: bool = True,
            pow: float = None,
            center: int = None,
        ):
            # 0. Default height and width to unet
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor

            original_size = original_size or (height, width)
            target_size = target_size or (height, width)

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                prompt_2,
                height,
                width,
                callback_steps,
                negative_prompt,
                negative_prompt_2,
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
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
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
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

            # 7. Prepare added time ids & embeddings
            add_text_embeds = pooled_prompt_embeds
            add_time_ids = self._get_add_time_ids(
                original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
            )

            if negative_original_size is not None and negative_target_size is not None:
                negative_add_time_ids = self._get_add_time_ids(
                    negative_original_size,
                    negative_crops_coords_top_left,
                    negative_target_size,
                    dtype=prompt_embeds.dtype,
                )
            else:
                negative_add_time_ids = add_time_ids

            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

            prompt_embeds = prompt_embeds.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

            # 8. Denoising loop
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

            # 7.1 Apply denoising_end
            if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
                discrete_timestep_cutoff = int(
                    round(
                        self.scheduler.config.num_train_timesteps
                        - (denoising_end * self.scheduler.config.num_train_timesteps)
                    )
                )
                num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
                timesteps = timesteps[:num_inference_steps]
            
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
                    )
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                    if i in interval_seq or cache_interval == 1:
                        prv_features = None
                        # print(t, prv_features is None)
                        # predict the noise residual
                        noise_pred, prv_features = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
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
                            cross_attention_kwargs=cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
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
                if self.needs_upcasting:
                    if not self.vae_upcasted:
                        self.upcast_vae()
                    dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype
                    latents = latents.to(dtype)

                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            else:
                image = latents

            if not output_type == "latent":
                # apply watermark if available
                if self.watermark is not None:
                    image = self.watermark.apply_watermark(image)

                image = self.image_processor.postprocess(image, output_type=output_type)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image,)

            return StableDiffusionXLPipelineOutput(images=image)
elif diffusers_version <= diffusers_0231_v:
    class StableDiffusionXLPipeline(DiffusersStableDiffusionXLPipeline):
        model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"
        _optional_components = ["tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2"]
        _callback_tensor_inputs = [
            "latents",
            "prompt_embeds",
            "negative_prompt_embeds",
            "add_text_embeds",
            "add_time_ids",
            "negative_pooled_prompt_embeds",
            "negative_add_time_ids",
        ]
        def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            text_encoder_2: CLIPTextModelWithProjection,
            tokenizer: CLIPTokenizer,
            tokenizer_2: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            force_zeros_for_empty_prompt: bool = True,
            add_watermarker: Optional[bool] = None,
        ):
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                unet=unet,
                scheduler=scheduler,
            )
            self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            self.default_sample_size = self.unet.config.sample_size

            add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

            self.fast_unet = FastUNet2DConditionModel(self.unet)

            self.vae_upcasted = False

            # make sure the VAE is in float32 mode, as it overflows in float16
            self.needs_upcasting = (
                self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            )

            if add_watermarker:
                self.watermark = StableDiffusionXLWatermarker()
            else:
                self.watermark = None
        
        def upcast_vae(self):
            super().upcast_vae()
            self.vae_upcasted = True
        
        @torch.no_grad()
        def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
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
                    "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
                )
            if callback_steps is not None:
                deprecate(
                    "callback_steps",
                    "1.0.0",
                    "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
                )
            # 0. Default height and width to unet
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor

            original_size = original_size or (height, width)
            target_size = target_size or (height, width)

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                prompt_2,
                height,
                width,
                callback_steps,
                negative_prompt,
                negative_prompt_2,
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
                callback_on_step_end_tensor_inputs,
            )

            self._guidance_scale = guidance_scale
            self._guidance_rescale = guidance_rescale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = cross_attention_kwargs
            self._denoising_end = denoising_end

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

            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
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

            # 7. Prepare added time ids & embeddings
            add_text_embeds = pooled_prompt_embeds
            if self.text_encoder_2 is None:
                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
            else:
                text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

            add_time_ids = self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            if negative_original_size is not None and negative_target_size is not None:
                negative_add_time_ids = self._get_add_time_ids(
                    negative_original_size,
                    negative_crops_coords_top_left,
                    negative_target_size,
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=text_encoder_projection_dim,
                )
            else:
                negative_add_time_ids = add_time_ids

            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

            prompt_embeds = prompt_embeds.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

            # 8. Denoising loop
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

            # 8.1 Apply denoising_end
            if (
                self.denoising_end is not None
                and isinstance(self.denoising_end, float)
                and self.denoising_end > 0
                and self.denoising_end < 1
            ):
                discrete_timestep_cutoff = int(
                    round(
                        self.scheduler.config.num_train_timesteps
                        - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                    )
                )
                num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
                timesteps = timesteps[:num_inference_steps]
            
            if diffusers_version > diffusers_0223_v:
                # 9. Optionally get Guidance Scale Embedding
                timestep_cond = None
                if self.unet.config.time_cond_proj_dim is not None:
                    guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                    timestep_cond = self.get_guidance_scale_embedding(
                        guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                    ).to(device=device, dtype=latents.dtype)
            
            self._num_timesteps = len(timesteps)
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
                    )
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

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
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                added_cond_kwargs=added_cond_kwargs,
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
                        add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                        negative_pooled_prompt_embeds = callback_outputs.pop(
                            "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                        )
                        add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                        negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

                    if XLA_AVAILABLE:
                        xm.mark_step()

            if not output_type == "latent":
                if self.needs_upcasting:
                    if not self.vae_upcasted:
                        self.upcast_vae()
                    dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype
                    latents = latents.to(dtype)

                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            else:
                image = latents

            if not output_type == "latent":
                # apply watermark if available
                if self.watermark is not None:
                    image = self.watermark.apply_watermark(image)

                image = self.image_processor.postprocess(image, output_type=output_type)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image,)

            return StableDiffusionXLPipelineOutput(images=image)

elif diffusers_version < diffusers_0270_v:
    class StableDiffusionXLPipeline(DiffusersStableDiffusionXLPipeline):
        if diffusers_version > diffusers_0240_v:
            model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
        else:
            model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"

        _optional_components = [
            "tokenizer",
            "tokenizer_2",
            "text_encoder",
            "text_encoder_2",
            "image_encoder",
            "feature_extractor",
        ]
        _callback_tensor_inputs = [
            "latents",
            "prompt_embeds",
            "negative_prompt_embeds",
            "add_text_embeds",
            "add_time_ids",
            "negative_pooled_prompt_embeds",
            "negative_add_time_ids",
        ]
        def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            text_encoder_2: CLIPTextModelWithProjection,
            tokenizer: CLIPTokenizer,
            tokenizer_2: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            image_encoder: CLIPVisionModelWithProjection = None,
            feature_extractor: CLIPImageProcessor = None,
            force_zeros_for_empty_prompt: bool = True,
            add_watermarker: Optional[bool] = None,
        ):
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                unet=unet,
                scheduler=scheduler,
                image_encoder=image_encoder,
                feature_extractor=feature_extractor,
            )
            self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            self.default_sample_size = self.unet.config.sample_size

            add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

            self.fast_unet = FastUNet2DConditionModel(self.unet)

            self.vae_upcasted = False

            # make sure the VAE is in float32 mode, as it overflows in float16
            self.needs_upcasting = (
                self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            )

            if add_watermarker:
                self.watermark = StableDiffusionXLWatermarker()
            else:
                self.watermark = None
        
        def upcast_vae(self):
            super().upcast_vae()
            self.vae_upcasted = True
        
        @torch.no_grad()
        def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
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
                    "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
                )
            if callback_steps is not None:
                deprecate(
                    "callback_steps",
                    "1.0.0",
                    "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
                )
            # 0. Default height and width to unet
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor

            original_size = original_size or (height, width)
            target_size = target_size or (height, width)

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                prompt_2,
                height,
                width,
                callback_steps,
                negative_prompt,
                negative_prompt_2,
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
                callback_on_step_end_tensor_inputs,
            )

            self._guidance_scale = guidance_scale
            self._guidance_rescale = guidance_rescale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = cross_attention_kwargs
            self._denoising_end = denoising_end
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

            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )

            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

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

            # 7. Prepare added time ids & embeddings
            add_text_embeds = pooled_prompt_embeds
            if self.text_encoder_2 is None:
                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
            else:
                text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

            add_time_ids = self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            if negative_original_size is not None and negative_target_size is not None:
                negative_add_time_ids = self._get_add_time_ids(
                    negative_original_size,
                    negative_crops_coords_top_left,
                    negative_target_size,
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=text_encoder_projection_dim,
                )
            else:
                negative_add_time_ids = add_time_ids

            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

            prompt_embeds = prompt_embeds.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

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
                        image_embeds = image_embeds.to(device)
            
            # 8. Denoising loop
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

            # 8.1 Apply denoising_end
            if (
                self.denoising_end is not None
                and isinstance(self.denoising_end, float)
                and self.denoising_end > 0
                and self.denoising_end < 1
            ):
                discrete_timestep_cutoff = int(
                    round(
                        self.scheduler.config.num_train_timesteps
                        - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                    )
                )
                num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
                timesteps = timesteps[:num_inference_steps]
            
            if diffusers_version > diffusers_0223_v:
                # 9. Optionally get Guidance Scale Embedding
                timestep_cond = None
                if self.unet.config.time_cond_proj_dim is not None:
                    guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                    timestep_cond = self.get_guidance_scale_embedding(
                        guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                    ).to(device=device, dtype=latents.dtype)
            
            self._num_timesteps = len(timesteps)
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
                    )
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if diffusers_version > diffusers_0240_v:
                        if self.interrupt:
                            continue

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    if ip_adapter_image is not None:
                        added_cond_kwargs["image_embeds"] = image_embeds

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
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                added_cond_kwargs=added_cond_kwargs,
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
                        add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                        negative_pooled_prompt_embeds = callback_outputs.pop(
                            "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                        )
                        add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                        negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

                    if XLA_AVAILABLE:
                        xm.mark_step()

            if not output_type == "latent":
                if self.needs_upcasting:
                    if not self.vae_upcasted:
                        self.upcast_vae()
                    dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype
                    latents = latents.to(dtype)

                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            else:
                image = latents

            if not output_type == "latent":
                # apply watermark if available
                if self.watermark is not None:
                    image = self.watermark.apply_watermark(image)

                image = self.image_processor.postprocess(image, output_type=output_type)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image,)

            return StableDiffusionXLPipelineOutput(images=image)
else:
    class StableDiffusionXLPipeline(DiffusersStableDiffusionXLPipeline):
        model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"

        _optional_components = [
            "tokenizer",
            "tokenizer_2",
            "text_encoder",
            "text_encoder_2",
            "image_encoder",
            "feature_extractor",
        ]
        _callback_tensor_inputs = [
            "latents",
            "prompt_embeds",
            "negative_prompt_embeds",
            "add_text_embeds",
            "add_time_ids",
            "negative_pooled_prompt_embeds",
            "negative_add_time_ids",
        ]
        def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            text_encoder_2: CLIPTextModelWithProjection,
            tokenizer: CLIPTokenizer,
            tokenizer_2: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            image_encoder: CLIPVisionModelWithProjection = None,
            feature_extractor: CLIPImageProcessor = None,
            force_zeros_for_empty_prompt: bool = True,
            add_watermarker: Optional[bool] = None,
        ):
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                unet=unet,
                scheduler=scheduler,
                image_encoder=image_encoder,
                feature_extractor=feature_extractor,
            )
            self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            self.default_sample_size = self.unet.config.sample_size

            add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

            self.fast_unet = FastUNet2DConditionModel(self.unet)

            self.vae_upcasted = False

            # make sure the VAE is in float32 mode, as it overflows in float16
            self.needs_upcasting = (
                self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            )

            if add_watermarker:
                self.watermark = StableDiffusionXLWatermarker()
            else:
                self.watermark = None
        
        def upcast_vae(self):
            super().upcast_vae()
            self.vae_upcasted = True
        
        @torch.no_grad()
        def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
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
                    "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
                )
            if callback_steps is not None:
                deprecate(
                    "callback_steps",
                    "1.0.0",
                    "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
                )
            # 0. Default height and width to unet
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor

            original_size = original_size or (height, width)
            target_size = target_size or (height, width)

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                prompt_2,
                height,
                width,
                callback_steps,
                negative_prompt,
                negative_prompt_2,
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
                ip_adapter_image,
                ip_adapter_image_embeds,
                callback_on_step_end_tensor_inputs,
            )

            self._guidance_scale = guidance_scale
            self._guidance_rescale = guidance_rescale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = cross_attention_kwargs
            self._denoising_end = denoising_end
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

            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )

            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

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

            # 7. Prepare added time ids & embeddings
            add_text_embeds = pooled_prompt_embeds
            if self.text_encoder_2 is None:
                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
            else:
                text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

            add_time_ids = self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            if negative_original_size is not None and negative_target_size is not None:
                negative_add_time_ids = self._get_add_time_ids(
                    negative_original_size,
                    negative_crops_coords_top_left,
                    negative_target_size,
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=text_encoder_projection_dim,
                )
            else:
                negative_add_time_ids = add_time_ids

            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

            prompt_embeds = prompt_embeds.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                image_embeds = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                    self.do_classifier_free_guidance,
                )
            
            # 8. Denoising loop
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

            # 8.1 Apply denoising_end
            if (
                self.denoising_end is not None
                and isinstance(self.denoising_end, float)
                and self.denoising_end > 0
                and self.denoising_end < 1
            ):
                discrete_timestep_cutoff = int(
                    round(
                        self.scheduler.config.num_train_timesteps
                        - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                    )
                )
                num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
                timesteps = timesteps[:num_inference_steps]
            
            # 9. Optionally get Guidance Scale Embedding
            timestep_cond = None
            if self.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                ).to(device=device, dtype=latents.dtype)
            
            self._num_timesteps = len(timesteps)
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
                    )
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                        added_cond_kwargs["image_embeds"] = image_embeds

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
                        add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                        negative_pooled_prompt_embeds = callback_outputs.pop(
                            "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                        )
                        add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                        negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

                    if XLA_AVAILABLE:
                        xm.mark_step()

            if not output_type == "latent":
                if self.needs_upcasting:
                    if not self.vae_upcasted:
                        self.upcast_vae()
                    dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype
                    latents = latents.to(dtype)

                # unscale/denormalize the latents
                # denormalize with the mean and std if available and not None
                has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
                has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
                if has_latents_mean and has_latents_std:
                    latents_mean = (
                        torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                    )
                    latents_std = (
                        torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                    )
                    latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
                else:
                    latents = latents / self.vae.config.scaling_factor

                image = self.vae.decode(latents, return_dict=False)[0]

            else:
                image = latents

            if not output_type == "latent":
                # apply watermark if available
                if self.watermark is not None:
                    image = self.watermark.apply_watermark(image)

                image = self.image_processor.postprocess(image, output_type=output_type)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image,)

            return StableDiffusionXLPipelineOutput(images=image)
