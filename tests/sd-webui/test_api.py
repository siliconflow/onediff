import base64
import io
import os
import urllib.request

import numpy as np
import pytest
import requests
import utils
import yaml
from PIL import Image
from utils import (IMG2IMG, IMG2IMG_API_ENDPOINT, NUM_STEPS, ONEDIFF,
                   ONEDIFF_QUANT, OPTIONS_API_ENDPOINT, SAVED_GRAPH_NAME, SEED,
                   TXT2IMG, TXT2IMG_API_ENDPOINT, base_prompt, height,
                   img2img_target_folder, txt2img_target_folder, width)

os.makedirs(img2img_target_folder, exist_ok=True)
os.makedirs(txt2img_target_folder, exist_ok=True)

keywords = ["onediff", "onediff-quant"]

# create target images if not exist
utils.check_and_generate_images(
    keywords, img2img_target_folder, txt2img_target_folder, width, height
)


@pytest.fixture()
def base_url():
    return f"http://127.0.0.1:7860"


@pytest.fixture()
def url_txt2img(base_url):
    return f"{base_url}/{TXT2IMG_API_ENDPOINT}"


@pytest.fixture()
def url_img2img(base_url):
    return f"{base_url}/{IMG2IMG_API_ENDPOINT}"


@pytest.fixture()
def url_set_config(base_url):
    return f"{base_url}/{OPTIONS_API_ENDPOINT}"


@pytest.fixture()
def simple_txt2img_request():
    return base_prompt


img_path = os.path.join(img2img_target_folder, "cat.png")
init_images = {"init_images": [utils.encode_file_to_base64(img_path)]}
data = {**base_prompt, **init_images}


test_cases = [
    ("utils.call_txt2img_api", "url_txt2img", base_prompt, ONEDIFF, TXT2IMG,),
    ("utils.call_txt2img_api", "url_txt2img", base_prompt, ONEDIFF_QUANT, TXT2IMG,),
    ("utils.call_img2img_api", "url_img2img", data, ONEDIFF, IMG2IMG,),
]


@pytest.mark.parametrize(
    "api, url_txt2img, request_payload, speed_up_method, image_gen_method",
    test_cases,
    indirect=["url_txt2img"],
)
def test_image_similarity_ssim(
    api, url_txt2img, request_payload, speed_up_method, image_gen_method
):
    api_function = eval(api)

    folder = (
        img2img_target_folder
        if image_gen_method == "img2img"
        else txt2img_target_folder
    )
    target_image_path = f"{folder}/{speed_up_method}-{image_gen_method}-w{width}-h{height}-seed-{SEED}-numstep-{NUM_STEPS}.png"

    generated_image = utils.send_request_and_get_image(
        api_function, url_txt2img, request_payload
    )
    target_image = np.array(Image.open(target_image_path))
    np_generated_image = utils.decode_image2array(generated_image)
    ssim_value = utils.cal_ssim(np_generated_image, target_image)
    print("SSIM: ", ssim_value)

    assert ssim_value > 0.99


def test_txt2img_onediff_save_graph(url_txt2img, simple_txt2img_request):
    script_args = {
        "script_args": [
            False,  # quantization
            None,  # graph_checkpoint
            SAVED_GRAPH_NAME,  # SAVED_GRAPH_NAME
        ]
    }
    data = {**simple_txt2img_request, **script_args}
    utils.post_request(url_txt2img, data)


def test_txt2img_onediff_load_graph(url_txt2img, simple_txt2img_request):
    script_args = {
        "script_args": [
            False,  # quantization
            SAVED_GRAPH_NAME,  # graph_checkpoint
            "",  # SAVED_GRAPH_NAME
        ]
    }
    data = {**simple_txt2img_request, **script_args}
    utils.post_request(url_txt2img, data)
