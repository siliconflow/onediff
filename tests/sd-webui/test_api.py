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

config = utils.read_config()


# init args
TXT2IMG_API_ENDPOINT = config["constants"]["TXT2IMG_API_ENDPOINT"]
IMG2IMG_API_ENDPOINT = config["constants"]["IMG2IMG_API_ENDPOINT"]
OPTIONS_API_ENDPOINT = config["constants"]["OPTIONS_API_ENDPOINT"]

HEIGHT = config["constants"]["HEIGHT"]
WIDTH = config["constants"]["WIDTH"]
SEED = config["constants"]["SEED"]
NUM_STEPS = config["constants"]["NUM_STEPS"]
img2img_target_folder = config["constants"]["IMG2IMG_TARGET_FOLDER"]
txt2img_target_folder = config["constants"]["TXT2IMG_TARGET_FOLDER"]
CFG_SCALE = config["constants"]["CFG_SCALE"]
N_ITER = config["constants"]["N_ITER"]
BATCH_SIZE = config["constants"]["BATCH_SIZE"]
ONEDIFF_QUANT = config["constants"]["ONEDIFF_QUANT"]
ONEDIFF = config["constants"]["ONEDIFF"]
IMG2IMG = config["constants"]["IMG2IMG"]
TXT2IMG = config["constants"]["TXT2IMG"]

os.makedirs(img2img_target_folder, exist_ok=True)
os.makedirs(txt2img_target_folder, exist_ok=True)

keywords = ["onediff", "onediff-quant"]
saved_graph_name = config["constants"]["SAVED_GRAPH_NAME"]

# create target images if not exist
utils.check_and_generate_images(
    keywords, img2img_target_folder, txt2img_target_folder, WIDTH, HEIGHT
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
    return {
        "prompt": "1girl",
        "negative_prompt": "",
        "seed": SEED,
        "steps": NUM_STEPS,
        "width": WIDTH,
        "height": HEIGHT,
        "cfg_scale": CFG_SCALE,
        "n_iter": N_ITER,
        "batch_size": BATCH_SIZE,
        # Enable OneDiff speed up
        "script_name": "onediff_diffusion_model",
        "script_args": [
            False,  # quantization
            None,  # graph_checkpoint
            "",  # saved_graph_name
        ],
    }


def test_txt2img_onediff(url_txt2img, simple_txt2img_request):
    data = simple_txt2img_request
    utils.post_request(url_txt2img, data)
    response = utils.call_txt2img_api(data)
    image = response.get("images")[0]

    target_image = np.array(
        Image.open(
            f"{txt2img_target_folder}/{ONEDIFF}-{TXT2IMG}-w{WIDTH}-h{HEIGHT}-seed-{SEED}-numstep-{NUM_STEPS}.png"
        )
    )
    imgdata = base64.b64decode(image)
    npimage = np.array(Image.open(io.BytesIO(imgdata)))

    ssim = utils.cal_ssim(npimage, target_image)
    assert ssim > 0.99


def test_img2img_onediff(url_img2img, simple_txt2img_request):
    img_path = os.path.join(img2img_target_folder, "cat.png")
    init_images = {"init_images": [utils.encode_file_to_base64(img_path)]}
    data = {**simple_txt2img_request, **init_images}

    utils.post_request(url_img2img, data)
    response = utils.call_img2img_api(data)

    image = response.get("images")[0]
    target_image = np.array(
        Image.open(
            f"{img2img_target_folder}/{ONEDIFF}-{IMG2IMG}-w{WIDTH}-h{HEIGHT}-seed-{SEED}-numstep-{NUM_STEPS}.png"
        )
    )
    imgdata = base64.b64decode(image)
    npimage = np.array(Image.open(io.BytesIO(imgdata)))

    ssim = utils.cal_ssim(npimage, target_image)
    assert ssim > 0.99


def test_txt2img_onediff_quant(url_txt2img, simple_txt2img_request):
    script_args = {
        "script_args": [
            True,  # quantization
            None,  # graph_checkpoint
            saved_graph_name,  # saved_graph_name
        ]
    }
    data = {**simple_txt2img_request, **script_args}

    utils.post_request(url_txt2img, data)
    response = utils.call_txt2img_api(data)

    target_image = np.array(
        Image.open(
            f"{txt2img_target_folder}/{ONEDIFF_QUANT}-{TXT2IMG}-w{WIDTH}-h{HEIGHT}-seed-{SEED}-numstep-{NUM_STEPS}.png"
        )
    )
    image = response.get("images")[0]

    imgdata = base64.b64decode(image)
    npimage = np.array(Image.open(io.BytesIO(imgdata)))

    ssim = utils.cal_ssim(npimage, target_image)
    assert ssim > 0.99


def test_txt2img_onediff_save_graph(url_txt2img, simple_txt2img_request):
    script_args = {
        "script_args": [
            False,  # quantization
            None,  # graph_checkpoint
            saved_graph_name,  # saved_graph_name
        ]
    }
    data = {**simple_txt2img_request, **script_args}
    utils.post_request(url_txt2img, data)


def test_txt2img_onediff_load_graph(url_txt2img, simple_txt2img_request):
    script_args = {
        "script_args": [
            False,  # quantization
            saved_graph_name,  # graph_checkpoint
            "",  # saved_graph_name
        ]
    }
    data = {**simple_txt2img_request, **script_args}
    utils.post_request(url_txt2img, data)
