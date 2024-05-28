import base64
import json
import os
import urllib.request
from pathlib import Path

import requests
from skimage.metrics import structural_similarity as ssim

webui_server_url = "http://127.0.0.1:7860"
base_prompt = {
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
TXT2IMG_API_ENDPOINT = "sdapi/v1/txt2img"
IMG2IMG_API_ENDPOINT = "sdapi/v1/img2img"


def check_and_generate_images(
    keywords, img2img_target_folder, txt2img_target_folder, WIDTH, HEIGHT
):
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


def txt2img_generate_imgs(save_path):
    payload_with_onediff = base_prompt
    response = call_txt2img_api(payload_with_onediff,)
    image = response.get("images")[0]
    decode_and_save_base64(
        image,
        os.path.join(save_path, "onediff-txt2img-w1024-h1024-seed-1-numstep-20.png"),
    )

    script_args = {
        "script_args": [
            True,  # quantization
            None,  # graph_checkpoint
            "saved_graph",  # saved_graph_name
        ]
    }
    payload_with_onediff_quant = {**payload_with_onediff, **script_args}
    response = call_txt2img_api(payload_with_onediff_quant,)
    image = response.get("images")[0]
    decode_and_save_base64(
        image,
        os.path.join(
            save_path, "onediff-quant-txt2img-w1024-h1024-seed-1-numstep-20.png"
        ),
    )


def img2img_generate_imgs(save_path):
    img_path = os.path.join(save_path, "cat.png")

    batch_size = 1
    init_images = {"init_images": [encode_file_to_base64(img_path)]}
    payload_with_onediff = {**base_prompt, **init_images}
    response = call_img2img_api(payload_with_onediff,)

    image = response.get("images")[0]
    decode_and_save_base64(
        image,
        os.path.join(save_path, "onediff-img2img-w1024-h1024-seed-1-numstep-20.png"),
    )

    script_args = {
        "script_args": [
            True,  # quantization
            None,  # graph_checkpoint
            "saved_graph",  # saved_graph_name
        ]
    }
    payload_with_onediff_quant = {**payload_with_onediff, **script_args}
    response = call_img2img_api(payload_with_onediff_quant,)
    image = response.get("images")[0]
    decode_and_save_base64(
        image,
        os.path.join(
            save_path, "onediff-quant-img2img-w1024-h1024-seed-1-numstep-20.png"
        ),
    )


def check_ssim(src, generated):
    ssim_score = ssim(src, generated, multichannel=True, win_size=3)
    print("SSIM:", ssim_score)
    return ssim_score
