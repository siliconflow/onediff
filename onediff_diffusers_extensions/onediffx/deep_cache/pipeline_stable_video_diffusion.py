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

import importlib.metadata
from typing import Callable, Dict, List, Optional, Union

from packaging import version

diffusers_0240_v = version.parse("0.24.0")
diffusers_0251_v = version.parse("0.25.1")
diffusers_0263_v = version.parse("0.26.3")
diffusers_0280_v = version.parse("0.28.0")
diffusers_version = version.parse(importlib.metadata.version("diffusers"))

import numpy as np
import PIL.Image
import torch


if diffusers_version >= diffusers_0280_v:
    from diffusers.video_processor import VideoProcessor
else:
    from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from diffusers import (
    StableVideoDiffusionPipeline as DiffusersStableVideoDiffusionPipeline,
)
from diffusers.pipelines.stable_video_diffusion import (
    StableVideoDiffusionPipelineOutput,
)

from .models.fast_unet_spatio_temporal_condition import (
    FastUNetSpatioTemporalConditionModel,
)

from .models.pipeline_utils import enable_deep_cache_pipeline

from .models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel


enable_deep_cache_pipeline()


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    # Based on:
    # https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78

    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    return outputs


class StableVideoDiffusionPipeline(DiffusersStableVideoDiffusionPipeline):
    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        if diffusers_version >= diffusers_0280_v:
            self.video_processor = VideoProcessor(
                do_resize=True, vae_scale_factor=self.vae_scale_factor
            )
        else:
            self.image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor
            )

        self.fast_unet = FastUNetSpatioTemporalConditionModel(self.unet)

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: float = 127,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        cache_interval: Optional[int] = 1,
        cache_branch: Optional[int] = None,
        return_dict: bool = True,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = (
            num_frames if num_frames is not None else self.unet.config.num_frames
        )
        decode_chunk_size = (
            decode_chunk_size if decode_chunk_size is not None else num_frames
        )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        if diffusers_version > diffusers_0240_v:
            self._guidance_scale = max_guidance_scale
        else:
            do_classifier_free_guidance = max_guidance_scale > 1.0

        # 3. Encode input image
        if diffusers_version > diffusers_0240_v:
            image_embeddings = self._encode_image(
                image, device, num_videos_per_prompt, self.do_classifier_free_guidance
            )
        else:
            image_embeddings = self._encode_image(
                image, device, num_videos_per_prompt, do_classifier_free_guidance
            )

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        if diffusers_version <= diffusers_0251_v:
            image = self.image_processor.preprocess(image, height=height, width=width)
            noise = randn_tensor(
                image.shape, generator=generator, device=image.device, dtype=image.dtype
            )
        elif diffusers_version < diffusers_0280_v:
            image = self.image_processor.preprocess(
                image, height=height, width=width
            ).to(device)
            noise = randn_tensor(
                image.shape, generator=generator, device=device, dtype=image.dtype
            )
        else:
            image = self.video_processor.preprocess(
                image, height=height, width=width
            ).to(device)
            noise = randn_tensor(
                image.shape, generator=generator, device=device, dtype=image.dtype
            )

        image = image + noise_aug_strength * noise

        needs_upcasting = (
            self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        )
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        if diffusers_version > diffusers_0251_v:
            image_latents = self._encode_vae_image(
                image,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )
        if diffusers_version > diffusers_0240_v:
            image_latents = self._encode_vae_image(
                image, device, num_videos_per_prompt, self.do_classifier_free_guidance
            )
        else:
            image_latents = self._encode_vae_image(
                image, device, num_videos_per_prompt, do_classifier_free_guidance
            )
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # 5. Get Added Time IDs
        if diffusers_version > diffusers_0240_v:
            added_time_ids = self._get_add_time_ids(
                fps,
                motion_bucket_id,
                noise_aug_strength,
                image_embeddings.dtype,
                batch_size,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
            )
        else:
            added_time_ids = self._get_add_time_ids(
                fps,
                motion_bucket_id,
                noise_aug_strength,
                image_embeddings.dtype,
                batch_size,
                num_videos_per_prompt,
                do_classifier_free_guidance,
            )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(
            min_guidance_scale, max_guidance_scale, num_frames
        ).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        cache_features = None
        interval_seq = list(range(0, num_inference_steps, cache_interval))
        interval_seq = sorted(interval_seq)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                if diffusers_version > diffusers_0240_v:
                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if self.do_classifier_free_guidance
                        else latents
                    )
                else:
                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if do_classifier_free_guidance
                        else latents
                    )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # Concatenate image_latents over channels dimention
                latent_model_input = torch.cat(
                    [latent_model_input, image_latents], dim=2
                )

                if i in interval_seq:
                    cache_features = None
                    # predict the noise residual
                    noise_pred, cache_features = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=image_embeddings,
                        added_time_ids=added_time_ids,
                        cache_features=cache_features,
                        cache_branch=cache_branch,
                        return_dict=False,
                    )
                else:
                    noise_pred, cache_features = self.fast_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=image_embeddings,
                        added_time_ids=added_time_ids,
                        cache_features=cache_features,
                        cache_branch=cache_branch,
                        return_dict=False,
                    )

                # perform guidance
                if diffusers_version > diffusers_0240_v:
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )
                else:
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            if diffusers_version < diffusers_0280_v:
                frames = tensor2vid(
                    frames, self.image_processor, output_type=output_type
                )
            else:
                frames = self.video_processor.postprocess_video(
                    video=frames, output_type=output_type
                )
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)
