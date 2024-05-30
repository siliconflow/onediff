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
from utils import (BATCH_SIZE, CFG_SCALE, IMG2IMG, IMG2IMG_API_ENDPOINT,
                   N_ITER, NUM_STEPS, ONEDIFF, ONEDIFF_QUANT,
                   OPTIONS_API_ENDPOINT, SAVED_GRAPH_NAME, SEED, TXT2IMG,
                   TXT2IMG_API_ENDPOINT, base_prompt, height,
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


def test_txt2img_onediff(url_txt2img, simple_txt2img_request):

    image = utils.send_request_and_get_image(
        utils.call_txt2img_api, url_txt2img, simple_txt2img_request
    )
    target_image = np.array(
        Image.open(
            f"{txt2img_target_folder}/{ONEDIFF}-{TXT2IMG}-w{width}-h{height}-seed-{SEED}-numstep-{NUM_STEPS}.png"
        )
    )
    # utils.decode_and_save_base64(
    #     image,
    #     f"{txt2img_target_folder}/{ONEDIFF}-{TXT2IMG}-w{width}-h{height}-seed-{SEED}-numstep-{NUM_STEPS}-ci.png",
    # )
    npimage = utils.decode_image2array(image)

    ssim = utils.cal_ssim(npimage, target_image)
    print("SSIM:", ssim)
    assert ssim > 0.99


def test_img2img_onediff(url_img2img, simple_txt2img_request):
    img_path = os.path.join(img2img_target_folder, "cat.png")
    init_images = {"init_images": [utils.encode_file_to_base64(img_path)]}
    data = {**simple_txt2img_request, **init_images}

    image = utils.send_request_and_get_image(utils.call_img2img_api, url_img2img, data)
    print(image)
    # utils.decode_and_save_base64(
    #     image,
    #     f"{img2img_target_folder}/{ONEDIFF}-{IMG2IMG}-w{width}-h{height}-seed-{SEED}-numstep-{NUM_STEPS}-ci.png",
    # )
    target_image = np.array(
        Image.open(
            f"{img2img_target_folder}/{ONEDIFF}-{IMG2IMG}-w{width}-h{height}-seed-{SEED}-numstep-{NUM_STEPS}.png"
        )
    )
    npimage = utils.decode_image2array(image)
    ssim = utils.cal_ssim(npimage, target_image)
    print("SSIM:", ssim)
    assert ssim > 0.99


def test_txt2img_onediff_quant(url_txt2img, simple_txt2img_request):
    script_args = {
        "script_args": [
            True,  # quantization
            None,  # graph_checkpoint
            SAVED_GRAPH_NAME,  # SAVED_GRAPH_NAME
        ]
    }
    data = {**simple_txt2img_request, **script_args}

    image = utils.send_request_and_get_image(utils.call_txt2img_api, url_txt2img, data)
    target_image = np.array(
        Image.open(
            f"{txt2img_target_folder}/{ONEDIFF_QUANT}-{TXT2IMG}-w{width}-h{height}-seed-{SEED}-numstep-{NUM_STEPS}.png"
        )
    )
    print(image)
    # utils.decode_and_save_base64(
    #     image,
    #     f"{txt2img_target_folder}/{ONEDIFF_QUANT}-{TXT2IMG}-w{width}-h{height}-seed-{SEED}-numstep-{NUM_STEPS}-ci.png",
    # )
    print(image)
    npimage = utils.decode_image2array(image)
    ssim = utils.cal_ssim(npimage, target_image)
    print("SSIM:", ssim)
    assert ssim > 0.99


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
