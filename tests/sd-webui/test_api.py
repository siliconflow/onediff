import pytest
import requests

@pytest.fixture()
def base_url():
    return f"http://127.0.0.1:7860"

@pytest.fixture()
def url_txt2img(base_url):
    return f"{base_url}/sdapi/v1/txt2img"

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
        # "script_name": "onediff_diffusion_model",

        # If you are using OneDiff Enterprise, add the field below to enable quant feature
        # "script_args" : [
        #     False, # quantization
        #     # None,  # graph_checkpoint
        #     # "",    # saved_graph_name
        # ],
    }

def test_txt2img_onediff(url_txt2img, simple_txt2img_request):
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200

def test_txt2img_onediff_quant(url_txt2img, simple_txt2img_request):
    # script_args = {
    #     "script_args": [
    #         True, # quantization
    #         #  None,  # graph_checkpoint
    #         #  "saved_graph",    # saved_graph_name
    #     ]
    # }
    # data = {**simple_txt2img_request, **script_args}
    # assert requests.post(url_txt2img, json=data).status_code == 200
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200

@pytest.mark.skip
def test_txt2img_onediff_save_graph(url_txt2img, simple_txt2img_request):
    script_args = {
        "script_args": [
            False, # quantization
            #  None,  # graph_checkpoint
            #  "saved_graph",    # saved_graph_name
        ]
    }
    data = {**simple_txt2img_request, **script_args}
    assert requests.post(url_txt2img, json=data).status_code == 200

@pytest.mark.skip
def test_txt2img_onediff_load_graph(url_txt2img, simple_txt2img_request):
    script_args = {
        "script_args": [
            False, # quantization
            #  None,  # graph_checkpoint
            #  "saved_graph",    # saved_graph_name
        ]
    }
    data = {**simple_txt2img_request, **script_args}
    assert requests.post(url_txt2img, json=data).status_code == 200
