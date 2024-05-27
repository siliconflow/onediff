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

webui_server_url = "http://127.0.0.1:7860"

out_dir = "api_out"
out_dir_t2i = os.path.join(out_dir, "txt2img")
os.makedirs(out_dir_t2i, exist_ok=True)


def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


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


def call_txt2img_api(**payload):
    response = call_api("sdapi/v1/txt2img", **payload)
    for index, image in enumerate(response.get("images")):
        save_path = os.path.join(out_dir_t2i, f"txt2img-{timestamp()}-{index}.png")
        decode_and_save_base64(image, save_path)


if __name__ == "__main__":
    payload = {
        "prompt": "masterpiece, (best quality:1.1), 1girl",
        # "prompt": "masterpiece, (best quality:1.1), 1girl <lora:lora_model:1>",  # extra networks also in prompts
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
        # If you are using OneDiff Enterprise, add the field below to enable quant feature
        # "script_args" : [{"0": True}],
    }
    call_txt2img_api(**payload)
