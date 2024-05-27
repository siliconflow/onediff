import base64
import pytest
import requests
from pathlib import Path
from utils import *
import urllib.request
from PIL import Image
import numpy as np
import io

img2img_target_folder = "/share_nfs/onediff_ci/sd-webui/images/img2img"
txt2img_target_folder = "/share_nfs/onediff_ci/sd-webui/images/txt2img"
os.makedirs(img2img_target_folder, exist_ok=True)
os.makedirs(txt2img_target_folder, exist_ok=True)
HEIGHT = 1024
WIDTH = 1024
keywords = ["onediff", "onediff-quant"]

# create target images if not exist
img2img_target_onediff_images = [
    f"{img2img_target_folder}/{keyword}-img2img-w{WIDTH}-h{HEIGHT}-seed-1-numstep-20.png"
    for keyword in keywords
]
txt2img_target_onediff_images = [
    f"{txt2img_target_folder}/{keyword}-txt2img-w{WIDTH}-h{HEIGHT}-seed-1-numstep-20.png"
    for keyword in keywords
]
if not all(Path(x).exists() for x in txt2img_target_onediff_images):
    print("Didn't find target txt2img images, try to generate...")
    txt2img_generate_imgs(txt2img_target_folder)

if not all(Path(x).exists() for x in img2img_target_onediff_images):
    print("Didn't find target img2img images, try to generate...")
    img2img_generate_imgs(img2img_target_folder)


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
    # response = post_request(url_txt2img, data)
    response = call_txt2img_api(data)
    image = response.get("images")[0]

    target_image = np.array(
        Image.open(
            f"{txt2img_target_folder}/onediff-txt2img-w{WIDTH}-h{HEIGHT}-seed-1-numstep-20.png"
        )
    )
    imgdata = base64.b64decode(image)
    npimage = np.array(Image.open(io.BytesIO(imgdata)))
    print(f"target_image {target_image.shape}")
    print(f"npimage {npimage.shape}")

    ssim = ssim_judge(npimage, target_image)
    print(f"txt2img_onediff ssim {ssim}")
    assert ssim > 0.94


def test_img2img_onediff(url_img2img, simple_txt2img_request):
    img_path = os.path.join(img2img_target_folder, "cat.png")
    init_images = {"init_images": [encode_file_to_base64(img_path)]}
    data = {**simple_txt2img_request, **init_images}
    # response = post_request(url_img2img, data)
    response = call_img2img_api(data)

    image = response.get("images")[0]

    target_image = np.array(
        Image.open(
            f"{img2img_target_folder}/onediff-img2img-w{WIDTH}-h{HEIGHT}-seed-1-numstep-20.png"
        )
    )
    imgdata = base64.b64decode(image)
    npimage = np.array(Image.open(io.BytesIO(imgdata)))
    ssim = ssim_judge(npimage, target_image)
    print(f"txt2img_onediff ssim {ssim}")
    assert ssim > 0.94


def test_txt2img_onediff_quant(url_txt2img, simple_txt2img_request):
    script_args = {
        "script_args": [
            True,  # quantization
            None,  # graph_checkpoint
            "saved_graph",  # saved_graph_name
        ]
    }
    data = {**simple_txt2img_request, **script_args}

    # response = post_request(url_txt2img, data)
    response = call_txt2img_api(data)

    target_image = np.array(
        Image.open(
            f"{txt2img_target_folder}/onediff-quant-txt2img-w{WIDTH}-h{HEIGHT}-seed-1-numstep-20.png"
        )
    )
    image = response.get("images")[0]

    imgdata = base64.b64decode(image)
    npimage = np.array(Image.open(io.BytesIO(imgdata)))
    ssim = ssim_judge(npimage, target_image)
    print(f"txt2img_onediff_quant ssim {ssim}")
    assert ssim > 0.94


def test_txt2img_onediff_save_graph(url_txt2img, simple_txt2img_request):
    script_args = {
        "script_args": [
            False,  # quantization
            None,  # graph_checkpoint
            "saved_graph",  # saved_graph_name
        ]
    }
    data = {**simple_txt2img_request, **script_args}
    post_request(url_txt2img, data)


def test_txt2img_onediff_load_graph(url_txt2img, simple_txt2img_request):
    script_args = {
        "script_args": [
            False,  # quantization
            "saved_graph",  # graph_checkpoint
            "",  # saved_graph_name
        ]
    }
    data = {**simple_txt2img_request, **script_args}
    post_request(url_txt2img, data)
