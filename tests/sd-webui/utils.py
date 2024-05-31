import base64
import io
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import requests
import yaml
from PIL import Image
from skimage.metrics import structural_similarity as ssim

TXT2IMG_API_ENDPOINT = "sdapi/v1/txt2img"
IMG2IMG_API_ENDPOINT = "sdapi/v1/img2img"
OPTIONS_API_ENDPOINT = "sdapi/v1/options"
IMG2IMG = "img2img"
TXT2IMG = "txt2img"
ONEDIFF_QUANT = "onediff-quant"
ONEDIFF = "onediff"
SEED = 1
NUM_STEPS = 20
height = 1024
width = 1024
img2img_target_folder = "/share_nfs/onediff_ci/sd-webui/images/img2img"
txt2img_target_folder = "/share_nfs/onediff_ci/sd-webui/images/txt2img"
SAVED_GRAPH_NAME = "saved_graph"
CFG_SCALE = 7
N_ITER = 1
BATCH_SIZE = 1


webui_server_url = "http://127.0.0.1:7860"


base_prompt = {
    "prompt": "1girl",
    "negative_prompt": "",
    "seed": SEED,
    "steps": NUM_STEPS,
    "width": width,
    "height": height,
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


def check_and_generate_images(
    keywords, img2img_target_folder, txt2img_target_folder, width, height
):
    img2img_target_onediff_images = [
        f"{img2img_target_folder}/{keyword}-{IMG2IMG}-w{width}-h{height}-seed-{SEED}-numstep-{NUM_STEPS}.png"
        for keyword in keywords
    ]
    txt2img_target_onediff_images = [
        f"{txt2img_target_folder}/{keyword}-{TXT2IMG}-w{width}-h{height}-seed-{SEED}-numstep-{NUM_STEPS}.png"
        for keyword in keywords
    ]

    if not all(Path(x).exists() for x in txt2img_target_onediff_images):
        txt2img_generate_onediff_imgs(txt2img_target_folder)
        txt2img_generate_onediff_quant_imgs(txt2img_target_folder)

    if not all(Path(x).exists() for x in img2img_target_onediff_images):
        img2img_generate_onediff_imgs(img2img_target_folder)
        img2img_generate_onediff_quant_imgs(img2img_target_folder)


def encode_file_to_base64(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def post_request(url, data):
    response = requests.post(url, json=data)
    assert response.status_code == 200


def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def call_api(api_endpoint, **payload):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{webui_server_url}/{api_endpoint}",
        headers={"Content-Type": "application/json"},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode("utf-8"))


def call_img2img_api(payload):
    response = call_api(IMG2IMG_API_ENDPOINT, **payload)
    return response


def call_txt2img_api(payload):
    response = call_api(TXT2IMG_API_ENDPOINT, **payload)
    return response


def txt2img_generate_onediff_imgs(save_path):
    payload_with_onediff = base_prompt
    response = call_txt2img_api(payload_with_onediff,)
    image = response.get("images")[0]
    decode_and_save_base64(
        image,
        os.path.join(
            save_path,
            f"{ONEDIFF}-{TXT2IMG}-w{width}-h{height}-seed-{SEED}-numstep-{NUM_STEPS}.png",
        ),
    )


def txt2img_generate_onediff_quant_imgs(save_path):
    script_args = {
        "script_args": [
            True,  # quantization
            None,  # graph_checkpoint
            SAVED_GRAPH_NAME,  # saved_graph_name
        ]
    }
    payload_with_onediff_quant = {**base_prompt, **script_args}
    response = call_txt2img_api(payload_with_onediff_quant,)
    image = response.get("images")[0]
    decode_and_save_base64(
        image,
        os.path.join(
            save_path,
            f"{ONEDIFF_QUANT}-{TXT2IMG}-w{width}-h{height}-seed-{SEED}-numstep-{NUM_STEPS}.png",
        ),
    )


def img2img_generate_onediff_imgs(save_path):
    img_path = os.path.join(save_path, "cat.png")

    init_images = {"init_images": [encode_file_to_base64(img_path)]}
    payload_with_onediff = {**base_prompt, **init_images}
    response = call_img2img_api(payload_with_onediff,)

    image = response.get("images")[0]
    decode_and_save_base64(
        image,
        os.path.join(
            save_path,
            f"{ONEDIFF}-{IMG2IMG}-w{width}-h{height}-seed-{SEED}-numstep-{NUM_STEPS}.png",
        ),
    )


def img2img_generate_onediff_quant_imgs(save_path):
    script_args = {
        "script_args": [
            True,  # quantization
            None,  # graph_checkpoint
            SAVED_GRAPH_NAME,  # SAVED_GRAPH_NAME
        ]
    }
    img_path = os.path.join(save_path, "cat.png")
    init_images = {"init_images": [encode_file_to_base64(img_path)]}
    payload_with_onediff = {**base_prompt, **init_images}
    payload_with_onediff_quant = {**payload_with_onediff, **script_args}
    response = call_img2img_api(payload_with_onediff_quant,)
    image = response.get("images")[0]
    decode_and_save_base64(
        image,
        os.path.join(
            save_path,
            f"{ONEDIFF_QUANT}-{IMG2IMG}-w{width}-h{height}-seed-{SEED}-numstep-{NUM_STEPS}.png",
        ),
    )


def cal_ssim(src, generated):
    ssim_score = ssim(src, generated, multichannel=True, win_size=3)
    return ssim_score


def send_request_and_get_image(api_call_func, url, data):
    post_request(url, data)
    response = api_call_func(data)
    return response.get("images")[0]


def decode_image2array(base64_string):
    imgdata = base64.b64decode(base64_string)
    npimage = np.array(Image.open(io.BytesIO(imgdata)))
    return npimage
