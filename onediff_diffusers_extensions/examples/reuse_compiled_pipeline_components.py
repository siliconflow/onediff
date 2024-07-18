"""
This example shows how to reuse the compiled components of a pipeline to create new pipelines.

Usage:
    $ python reuse_compiled_pipeline_components.py --model_id <model_id>
"""
import argparse
from io import BytesIO

import PIL
import requests
import torch
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
from onediffx import compile_pipe


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    return parser.parse_args()


args = get_args()


def initialize_pipelines(model_id, model_params):
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, **model_params)

    pipeline.to("cuda")
    pipeline = compile_pipe(pipeline)

    # Reuse the components of the pipeline to create new pipelines
    # Reference: https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.components
    img2img_pipe = StableDiffusionImg2ImgPipeline(**pipeline.components)
    inpaint_pipe = StableDiffusionInpaintPipeline(**pipeline.components)
    return pipeline, img2img_pipe, inpaint_pipe


def download_image(url, resize=None):
    response = requests.get(url)
    image = PIL.Image.open(BytesIO(response.content)).convert("RGB")
    if resize:
        image = image.resize(resize)
    return image


def inference_text2img(pipe):
    prompt = "A fantasy landscape, trending on artstation"
    result_image = pipe(prompt=prompt).images[0]
    result_image.save("inference_text2img.png")


def inference_img2img(pipe):
    img_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    init_image = download_image(img_url, resize=(768, 512))
    prompt = "A fantasy landscape, trending on artstation"
    result_image = pipe(
        prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5
    ).images[0]
    result_image.save("inference_img2img.png")


def inference_inpaint(pipe):
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    init_image = download_image(img_url, resize=(512, 512))
    mask_image = download_image(mask_url, resize=(512, 512))

    prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
    result_image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[
        0
    ]
    result_image.save("inference_inpaint.png")


def main(model_id, model_params):
    text2img, img2img, inpaint = initialize_pipelines(model_id, model_params)

    inference_text2img(text2img)
    inference_img2img(img2img)
    inference_inpaint(inpaint)


if __name__ == "__main__":
    model_id = args.model_id
    model_params = {"torch_dtype": torch.float16}
    main(model_id, model_params)
