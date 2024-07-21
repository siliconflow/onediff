import base64
import io
import os
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import requests
from PIL import Image
from skimage.metrics import structural_similarity as ssim

WEBUI_SERVER_URL = "http://127.0.0.1:7860"
TXT2IMG_API_ENDPOINT = "sdapi/v1/txt2img"
IMG2IMG_API_ENDPOINT = "sdapi/v1/img2img"
OPTIONS_API_ENDPOINT = "sdapi/v1/options"
SEED = 1
NUM_STEPS = 20
HEIGHT = 1024
WIDTH = 1024
IMG2IMG_TARGET_FOLDER = "/share_nfs/onediff_ci/sd-webui/images/img2img"
TXT2IMG_TARGET_FOLDER = "/share_nfs/onediff_ci/sd-webui/images/txt2img"
SAVED_GRAPH_NAME = "saved_graph"

os.makedirs(IMG2IMG_TARGET_FOLDER, exist_ok=True)
os.makedirs(TXT2IMG_TARGET_FOLDER, exist_ok=True)


def get_base_args() -> Dict[str, Any]:
    return {
        "prompt": "1girl",
        "negative_prompt": "",
        "seed": SEED,
        "steps": NUM_STEPS,
        "width": WIDTH,
        "height": HEIGHT,
        "cfg_scale": 7,
        "n_iter": 1,
        "batch_size": 1,
        # Enable OneDiff speed up
        "script_name": "onediff_diffusion_model",
    }


def get_extra_args() -> List[Dict[str, Any]]:
    quant_args = [
        {
            "script_args": [
                x,  # quantization
                None,  # graph_checkpoint
                "",  # saved_graph_name
            ]
        }
        for x in [True, False]
    ]

    txt2img_args = [
        {},
        {"init_images": [get_init_image()]},
    ]

    return [
        quant_args,
        txt2img_args,
    ]


def get_all_args() -> Iterable[Dict[str, Any]]:
    for extra_args in product(*get_extra_args()):
        args = get_base_args()
        for extra_arg in extra_args:
            args = {**args, **extra_arg}
        yield args


def is_txt2img(data: Dict[str, Any]) -> bool:
    return "init_images" not in data


def is_quant(data: Dict[str, Any]) -> bool:
    return data["script_args"][0]


def encode_file_to_base64(path: str) -> str:
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def post_request_and_check(url: str, data: Dict[str, Any]):
    response = requests.post(url, json=data)
    assert response.status_code == 200
    return response


def get_image_byte_from_response(response):
    return base64.b64decode(response.json()["images"][0])


def get_image_array_from_response(response):
    imgdata = base64.b64decode(response.json()["images"][0])
    return np.array(Image.open(io.BytesIO(imgdata)))


def get_init_image():
    img_path = str(Path(__file__).parent / "cat.png")
    return encode_file_to_base64(img_path)


def cal_ssim(src: np.ndarray, generated: np.ndarray) -> float:
    ssim_score = ssim(src, generated, multichannel=True, win_size=3)
    return ssim_score


def generate_image(filename: str, data: Dict[str, Any]):
    endpoint = TXT2IMG_API_ENDPOINT if is_txt2img(data) else IMG2IMG_API_ENDPOINT
    url = f"{WEBUI_SERVER_URL}/{endpoint}"
    response = post_request_and_check(url, data)
    image = get_image_byte_from_response(response)
    with open(filename, "wb") as file:
        file.write(image)


def get_target_image_filename(data: Dict[str, Any]) -> str:
    parent_path = TXT2IMG_TARGET_FOLDER if is_txt2img(data) else IMG2IMG_TARGET_FOLDER
    if not Path(parent_path).exists():
        Path(parent_path).mkdir(mode=777, parents=True)

    txt2img_str = "txt2img" if is_txt2img(data) else "img2img"
    quant_str = "-quant" if is_quant(data) else ""
    return f"{parent_path}/onediff{quant_str}-{txt2img_str}-w{WIDTH}-h{HEIGHT}-seed-{SEED}-numstep-{NUM_STEPS}.png"


def check_and_generate_images():
    for data in get_all_args():
        image_path = get_target_image_filename(data)
        if not Path(image_path).exists():
            print(f"Generating image for {get_data_summary(data)}...")
            generate_image(image_path, data)
        else:
            print(f"Image for {get_data_summary(data)} exists, skip generating...")


def get_data_summary(data: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "is_txt2img": is_txt2img(data),
        "is_quant": is_quant(data),
    }


def dump_image(src_img: np.ndarray, target_img: np.ndarray, filename: str):
    combined_img = np.concatenate((src_img, target_img), axis=1)
    image = Image.fromarray(combined_img)
    image.save(f"{filename}.png")


def get_threshold(data: Dict[str, Any]):
    if is_quant(data):
        return 0.7
    else:
        return 0.95
