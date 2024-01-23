from datetime import datetime
import urllib.request
import base64
import json
import time
import os

webui_server_url = 'http://127.0.0.1:7860'

out_dir = 'api_out'
out_dir_t2i = os.path.join(out_dir, 'txt2img')
out_dir_i2i = os.path.join(out_dir, 'img2img')
os.makedirs(out_dir_t2i, exist_ok=True)
os.makedirs(out_dir_i2i, exist_ok=True)


def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')


def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def call_api(api_endpoint, **payload):
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{webui_server_url}/{api_endpoint}',
        headers={'Content-Type': 'application/json'},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode('utf-8'))


def call_txt2img_api(**payload):
    response = call_api('sdapi/v1/txt2img', **payload)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir_t2i, f'txt2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)


def call_img2img_api(**payload):
    response = call_api('sdapi/v1/img2img', **payload)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir_i2i, f'img2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)


if __name__ == '__main__':
    payload = {
        "prompt": "masterpiece, (best quality:1.1), 1girl",
        # "prompt": "masterpiece, (best quality:1.1), 1girl <lora:lora_model:1>",  # extra networks also in prompts
        "negative_prompt": "",
        "seed": 1,
        "steps": 20,
        "width": 1024,
        "height": 1024,
        "cfg_scale": 7,
        "sampler_name": "DPM++ 2M Karras",
        "n_iter": 1,
        "batch_size": 1,

        # Enable OneDiff speed up
        "script_name": "onediff_diffusion_model",
        # If you are using OneDiff Enterprise, add the field below to enable quant feature
        # "script_args" : [{"0": True}],

        # example args for x/y/z plot
        # "script_name": "x/y/z plot",
        # "script_args": [
        #     1,
        #     "10,20",
        #     [],
        #     0,
        #     "",
        #     [],
        #     0,
        #     "",
        #     [],
        #     True,
        #     True,
        #     False,
        #     False,
        #     0,
        #     False
        # ],

        # example args for Refiner and ControlNet
        # "alwayson_scripts": {
        #     "ControlNet": {
        #         "args": [
        #             {
        #                 "batch_images": "",
        #                 "control_mode": "Balanced",
        #                 "enabled": True,
        #                 "guidance_end": 1,
        #                 "guidance_start": 0,
        #                 "image": {
        #                     "image": encode_file_to_base64(r"B:\path\to\control\img.png"),
        #                     "mask": None  # base64, None when not need
        #                 },
        #                 "input_mode": "simple",
        #                 "is_ui": True,
        #                 "loopback": False,
        #                 "low_vram": False,
        #                 "model": "control_v11p_sd15_canny [d14c016b]",
        #                 "module": "canny",
        #                 "output_dir": "",
        #                 "pixel_perfect": False,
        #                 "processor_res": 512,
        #                 "resize_mode": "Crop and Resize",
        #                 "threshold_a": 100,
        #                 "threshold_b": 200,
        #                 "weight": 1
        #             }
        #         ]
        #     },
        #     "Refiner": {
        #         "args": [
        #             True,
        #             "sd_xl_refiner_1.0",
        #             0.5
        #         ]
        #     }
        # },
        # "enable_hr": True,
        # "hr_upscaler": "R-ESRGAN 4x+ Anime6B",
        # "hr_scale": 2,
        # "denoising_strength": 0.5,
        # "styles": ['style 1', 'style 2'],
        # "override_settings": {
        #     'sd_model_checkpoint': "sd_xl_base_1.0",  # this can use to switch sd model
        # },
    }
    call_txt2img_api(**payload)

    # OneDiff sd-webui doesn't support img2img now, the feature is under urgent development
    # init_images = [
    #     encode_file_to_base64(r"B:\path\to\img_1.png"),
    #     # encode_file_to_base64(r"B:\path\to\img_2.png"),
    #     # "https://image.can/also/be/a/http/url.png",
    # ]

    # batch_size = 2
    # payload = {
    #     "prompt": "1girl, blue hair",
    #     "seed": 1,
    #     "steps": 20,
    #     "width": 512,
    #     "height": 512,
    #     "denoising_strength": 0.5,
    #     "n_iter": 1,
    #     "init_images": init_images,
    #     "batch_size": batch_size if len(init_images) == 1 else len(init_images),
    #     # "mask": encode_file_to_base64(r"B:\path\to\mask.png")
    # }
    # # if len(init_images) > 1 then batch_size should be == len(init_images)
    # # else if len(init_images) == 1 then batch_size can be any value int >= 1
    # call_img2img_api(**payload)

    # there exist a useful extension that allows converting of webui calls to api payload
    # particularly useful when you wish setup arguments of extensions and scripts
    # https://github.com/huchenlei/sd-webui-api-payload-display
