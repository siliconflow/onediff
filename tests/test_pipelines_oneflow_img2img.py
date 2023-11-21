# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

import gc
import random
import unittest

import numpy as np
import oneflow as torch

from onediff import OneFlowStableDiffusionImg2ImgPipeline

from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)

from diffusers.utils import floats_tensor, load_image, torch_device
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer


class PipelineFastTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def dummy_image(self):
        batch_size = 1
        num_channels = 3
        sizes = (32, 32)

        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0))
        return torch.tensor(image)

    @property
    def dummy_cond_unet(self):
        torch.manual_seed(0)
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        return model

    @property
    def dummy_vae(self):
        torch.manual_seed(0)
        model = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        return model

    @property
    def dummy_text_encoder(self):
        torch.manual_seed(0)
        config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        return CLIPTextModel(config)

    @property
    def dummy_safety_checker(self):
        def check(images, *args, **kwargs):
            return images, [False] * len(images)

        return check

    @property
    def dummy_extractor(self):
        def extract(*args, **kwargs):
            if "return_tensors" in kwargs:
                return_tensors = kwargs["return_tensors"]
            else:
                return_tensors = "pt"

            class Out:
                def __init__(self):
                    self.pixel_values = torch.ones([0])
                    if return_tensors == "np":
                        self.pixel_values = torch.ones([0]).numpy()

                def to(self, device):
                    if return_tensors == "np":
                        return self
                    self.pixel_values.to(device)
                    return self

            return Out()

        return extract

    def test_stable_diffusion_img2img(self):
        unet = self.dummy_cond_unet.to(torch_device)
        scheduler = PNDMScheduler(skip_prk_steps=True, steps_offset=1)
        vae = self.dummy_vae.to(torch_device)
        bert = self.dummy_text_encoder.to(torch_device)
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip"
        )

        init_image = self.dummy_image.to(torch_device)

        # make sure here that pndm scheduler skips prk
        sd_pipe = OneFlowStableDiffusionImg2ImgPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        # prompt = "sea,beach,the waves crashed on the sand,blue sky whit white cloud"
        generator = torch.Generator(device=torch_device).manual_seed(0)
        output = sd_pipe(
            [prompt],
            generator=generator,
            strength=0.75,
            guidance_scale=7.5,
            output_type="np",
            image=init_image,
            compile_unet=False,
        )
        image = output.images

        generator = torch.Generator(device=torch_device).manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            generator=generator,
            strength=0.75,
            guidance_scale=7.5,
            output_type="np",
            image=init_image,
            return_dict=False,
            compile_unet=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)

        # Do not modify any seed number to past this test
        expected_slice = np.array(
            [0.4287, 0.5450, 0.5239, 0.5432, 0.6519, 0.5665, 0.6027, 0.5805, 0.5145]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_img2img_k_lms(self):
        unet = self.dummy_cond_unet
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )

        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip"
        )
        init_image = self.dummy_image.to(torch_device)

        # make sure here that pndm scheduler skips prk
        sd_pipe = OneFlowStableDiffusionImg2ImgPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=self.dummy_safety_checker,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device=torch_device).manual_seed(0)
        output = sd_pipe(
            [prompt],
            image=init_image,
            generator=generator,
            strength=0.75,
            guidance_scale=7.5,
            output_type="np",
            compile_unet=False,
        )
        image = output.images

        generator = torch.Generator(device=torch_device).manual_seed(0)
        output = sd_pipe(
            [prompt],
            image=init_image,
            generator=generator,
            strength=0.75,
            guidance_scale=7.5,
            output_type="np",
            return_dict=False,
            compile_unet=False,
        )
        image_from_tuple = output[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)

        # Do not modify any seed number to past this test
        expected_slice = np.array(
            [0.4213, 0.5489, 0.5102, 0.5320, 0.6574, 0.5861, 0.6171, 0.5866, 0.5160]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_img2img_pipeline(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        )
        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/fantasy_landscape.png"
        )
        init_image = init_image.resize((768, 512))
        expected_image = np.array(expected_image, dtype=np.float32) / 255.0

        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = OneFlowStableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, safety_checker=self.dummy_safety_checker, use_auth_token=True,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A fantasy landscape, trending on artstation"

        generator = torch.Generator(device=torch_device).manual_seed(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            generator=generator,
            output_type="np",
            compile_unet=False,
        )
        image = output.images[0]

        assert image.shape == (512, 768, 3)
        # img2img is flaky across GPUs even in fp32, so using MAE here
        assert np.abs(expected_image - image).mean() < 1e-2

    def test_stable_diffusion_img2img_pipeline_k_lms(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        )
        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/fantasy_landscape_k_lms.png"
        )
        init_image = init_image.resize((768, 512))
        expected_image = np.array(expected_image, dtype=np.float32) / 255.0

        lms = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )

        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = OneFlowStableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            scheduler=lms,
            safety_checker=self.dummy_safety_checker,
            use_auth_token=True,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A fantasy landscape, trending on artstation"

        generator = torch.Generator(device=torch_device).manual_seed(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            generator=generator,
            output_type="np",
            compile_unet=False,
        )
        image = output.images[0]

        assert image.shape == (512, 768, 3)
        # img2img is flaky across GPUs even in fp32, so using MAE here
        assert np.abs(expected_image - image).mean() < 1e-2


if __name__ == "__main__":
    unittest.main()
