import base64
import json
import os
import urllib.request
from pathlib import Path

import requests
import yaml
from skimage.metrics import structural_similarity as ssim


def read_config(file_path=None):
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), "config.yml")
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


webui_server_url = "http://127.0.0.1:7860"

# init args
config = read_config()
TXT2IMG_API_ENDPOINT = config["constants"]["TXT2IMG_API_ENDPOINT"]
IMG2IMG_API_ENDPOINT = config["constants"]["IMG2IMG_API_ENDPOINT"]
IMG2IMG = config["constants"]["IMG2IMG"]
TXT2IMG = config["constants"]["TXT2IMG"]
HEIGHT = config["constants"]["HEIGHT"]
WIDTH = config["constants"]["WIDTH"]
ONEDIFF_QUANT = config["constants"]["ONEDIFF_QUANT"]
ONEDIFF = config["constants"]["ONEDIFF"]
SEED = config["constants"]["SEED"]
NUM_STEPS = config["constants"]["NUM_STEPS"]
CFG_SCALE = config["constants"]["CFG_SCALE"]
N_ITER = config["constants"]["N_ITER"]
BATCH_SIZE = config["constants"]["BATCH_SIZE"]
SAVED_GRAPH_NAME = config["constants"]["SAVED_GRAPH_NAME"]

base_prompt = {
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


def check_and_generate_images(
    keywords, img2img_target_folder, txt2img_target_folder, WIDTH, HEIGHT
):
    img2img_target_onediff_images = [
        f"{img2img_target_folder}/{keyword}-{IMG2IMG}-w{WIDTH}-h{HEIGHT}-seed-{SEED}-numstep-{NUM_STEPS}.png"
        for keyword in keywords
    ]
    txt2img_target_onediff_images = [
        f"{txt2img_target_folder}/{keyword}-{TXT2IMG}-w{WIDTH}-h{HEIGHT}-seed-{SEED}-numstep-{NUM_STEPS}.png"
        for keyword in keywords
    ]

    if not all(Path(x).exists() for x in txt2img_target_onediff_images):
        print("Didn't find target txt2img images, try to generate...")
        txt2img_generate_onediff_imgs(txt2img_target_folder)
        txt2img_generate_onediff_quant_imgs(txt2img_target_folder)

    if not all(Path(x).exists() for x in img2img_target_onediff_images):
        print("Didn't find target img2img images, try to generate...")
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
            "{ONEDIFF}-{TXT2IMG}-w{WIDTH}-h{HEIGHT}-seed-{SEED}-numstep-{NUM_STEPS}.png",
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
            "{ONEDIFF_QUANT}-{TXT2IMG}-w{WIDTH}-h{HEIGHT}-seed-{SEED}-numstep-{NUM_STEPS}.png",
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
            "{ONEDIFF}-{IMG2IMG}-w{WIDTH}-h{HEIGHT}-seed-{SEED}-numstep-{NUM_STEPS}.png",
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
    print(f"payload_with_onediff_quant {payload_with_onediff_quant}")
    response = call_img2img_api(payload_with_onediff_quant,)
    image = response.get("images")[0]
    decode_and_save_base64(
        image,
        os.path.join(
            save_path,
            "{ONEDIFF_QUANT}-{IMG2IMG}-w{WIDTH}-h{HEIGHT}-seed-{SEED}-numstep-{NUM_STEPS}.png",
        ),
    )


def cal_ssim(src, generated):
    ssim_score = ssim(src, generated, multichannel=True, win_size=3)
    print("SSIM:", ssim_score)
    return ssim_score
