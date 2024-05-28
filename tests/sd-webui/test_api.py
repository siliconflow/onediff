import base64
import io
import os
import urllib.request

import numpy as np
import pytest
import requests
import utils
from PIL import Image

img2img_target_folder = "/share_nfs/onediff_ci/sd-webui/images/img2img"
txt2img_target_folder = "/share_nfs/onediff_ci/sd-webui/images/txt2img"
os.makedirs(img2img_target_folder, exist_ok=True)
os.makedirs(txt2img_target_folder, exist_ok=True)
HEIGHT = 1024
WIDTH = 1024
keywords = ["onediff", "onediff-quant"]
saved_graph_name = "saved_graph"

# create target images if not exist
utils.check_and_generate_images(
    keywords, img2img_target_folder, txt2img_target_folder, WIDTH, HEIGHT
)


@pytest.fixture()
def base_url():
    return f"http://127.0.0.1:7860"


@pytest.fixture()
def url_txt2img(base_url):
    return f"{base_url}/sdapi/v1/txt2img"


@pytest.fixture()
def url_img2img(base_url):
    return f"{base_url}/sdapi/v1/img2img"


@pytest.fixture()
def url_set_config(base_url):
    return f"{base_url}/sdapi/v1/options"


@pytest.fixture()
def simple_txt2img_request():
    return {
        "prompt": "1girl",
        "negative_prompt": "",
        "seed": 1,
        "steps": 20,
        "width": 1024,
        "height": 1024,
        "cfg_scale": 7,
        "n_iter": 1,
        "batch_size": 1,
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
            f"{txt2img_target_folder}/onediff-txt2img-w{WIDTH}-h{HEIGHT}-seed-1-numstep-20.png"
        )
    )
    imgdata = base64.b64decode(image)
    npimage = np.array(Image.open(io.BytesIO(imgdata)))

    ssim = utils.check_ssim(npimage, target_image)
    assert ssim > 0.99


def test_img2img_onediff(url_img2img, simple_txt2img_request):
    img_path = os.path.join(img2img_target_folder, "cat.png")
    init_images = {"init_images": [utils.encode_file_to_base64(img_path)]}
    data = {**simple_txt2img_request, **init_images}
    utils.post_request(url_img2img, data)
    response = utils.call_txt2img_api(data)

    image = response.get("images")[0]

    target_image = np.array(
        Image.open(
            f"{img2img_target_folder}/onediff-img2img-w{WIDTH}-h{HEIGHT}-seed-1-numstep-20.png"
        )
    )
    imgdata = base64.b64decode(image)
    npimage = np.array(Image.open(io.BytesIO(imgdata)))
    ssim = utils.check_ssim(npimage, target_image)
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
            f"{txt2img_target_folder}/onediff-quant-txt2img-w{WIDTH}-h{HEIGHT}-seed-1-numstep-20.png"
        )
    )
    image = response.get("images")[0]

    imgdata = base64.b64decode(image)
    npimage = np.array(Image.open(io.BytesIO(imgdata)))
    ssim = utils.check_ssim(npimage, target_image)
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
