import base64
import pytest
import requests
from pathlib import Path

def encode_file_to_base64(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")

def post_request(url, data):
    response = requests.post(url, json=data)
    assert response.status_code == 200
    return response

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

        "script_args" : [
            False, # quantization
            None,  # graph_checkpoint
            "",    # saved_graph_name
        ],
    }

def test_txt2img_onediff(url_txt2img, simple_txt2img_request):
    data = simple_txt2img_request
    post_request(url_txt2img, data)

def test_img2img_onediff(url_img2img, simple_txt2img_request):
    img_path = str(Path(__file__).parent / "cat.png")
    init_images = {"init_images": [encode_file_to_base64(img_path)]}
    data = {**simple_txt2img_request, **init_images}
    post_request(url_img2img, data)

def test_txt2img_onediff_quant(url_txt2img, simple_txt2img_request):
    script_args = {
        "script_args": [
            True,           # quantization
            None,           # graph_checkpoint
            "saved_graph",  # saved_graph_name
        ]
    }
    data = {**simple_txt2img_request, **script_args}
    post_request(url_txt2img, data)

def test_txt2img_onediff_save_graph(url_txt2img, simple_txt2img_request):
    script_args = {
        "script_args": [
            False,          # quantization
            None,           # graph_checkpoint
            "saved_graph",  # saved_graph_name
        ]
    }
    data = {**simple_txt2img_request, **script_args}
    post_request(url_txt2img, data)

def test_txt2img_onediff_load_graph(url_txt2img, simple_txt2img_request):
    script_args = {
        "script_args": [
            False,          # quantization
            "saved_graph",  # graph_checkpoint
            "",             # saved_graph_name
        ]
    }
    data = {**simple_txt2img_request, **script_args}
    post_request(url_txt2img, data)
