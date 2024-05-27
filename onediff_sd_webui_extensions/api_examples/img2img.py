# The code is referenced from gists https://gist.github.com/w-e-w/0f37c04c18e14e4ee1482df5c4eb9f53 on
# Stable-Diffusion-WebUI API wiki https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API

# To use OneDiff to speed up model inference, you only need to
# add `"script_name": "onediff_diffusion_model"` to the data of the request
# And if you are using OneDiff Enterprise, add another
# `"script_args" : [{"0": True}]` field to enable quantization

import base64
import json
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path

webui_server_url = "http://127.0.0.1:7860"

out_dir = "api_out"
out_dir_i2i = os.path.join(out_dir, "img2img")
os.makedirs(out_dir_i2i, exist_ok=True)


def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def encode_file_to_base64(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


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


def call_img2img_api(**payload):
    response = call_api("sdapi/v1/img2img", **payload)
    for index, image in enumerate(response.get("images")):
        save_path = os.path.join(out_dir_i2i, f"img2img-{timestamp()}-{index}.png")
        decode_and_save_base64(image, save_path)


if __name__ == "__main__":
    img_path = str(Path(__file__).parent / "cat.png")
    init_images = [
        encode_file_to_base64(img_path),
        # encode_file_to_base64(r"B:\path\to\img_2.png"),
        # "https://image.can/also/be/a/http/url.png",
    ]

    # batch_size = 2
    batch_size = 1
    payload = {
        "prompt": "1girl, blue hair",
        "seed": 1,
        "steps": 20,
        "width": 1024,
        "height": 1024,
        "denoising_strength": 0.5,
        "n_iter": 1,
        "init_images": init_images,
        "batch_size": batch_size if len(init_images) == 1 else len(init_images),
        # Enable OneDiff speed up
        "script_name": "onediff_diffusion_model",
        # If you are using OneDiff Enterprise, add the field below to enable quant feature
        # "script_args" : [{"0": True}],
        # "mask": encode_file_to_base64(r"B:\path\to\mask.png")
    }
    # if len(init_images) > 1 then batch_size should be == len(init_images)
    # else if len(init_images) == 1 then batch_size can be any value int >= 1
    call_img2img_api(**payload)

    # there exist a useful extension that allows converting of webui calls to api payload
    # particularly useful when you wish setup arguments of extensions and scripts
    # https://github.com/huchenlei/sd-webui-api-payload-display
