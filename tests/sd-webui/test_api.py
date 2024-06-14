import os

import numpy as np
import pytest
from PIL import Image
from utils import (
    IMG2IMG_API_ENDPOINT,
    OPTIONS_API_ENDPOINT,
    SAVED_GRAPH_NAME,
    TXT2IMG_API_ENDPOINT,
    WEBUI_SERVER_URL,
    cal_ssim,
    check_and_generate_images,
    get_all_args,
    get_base_args,
    get_data_summary,
    get_image_array_from_response,
    get_target_image_filename,
    is_txt2img,
    post_request_and_check,
    save_image,
)


@pytest.fixture(scope="session", autouse=True)
def prepare_target_images():
    print("checking if target images exist...")
    check_and_generate_images()


@pytest.fixture()
def base_url():
    return WEBUI_SERVER_URL


@pytest.fixture()
def url_txt2img(base_url):
    return f"{base_url}/{TXT2IMG_API_ENDPOINT}"


@pytest.fixture()
def url_img2img(base_url):
    return f"{base_url}/{IMG2IMG_API_ENDPOINT}"


@pytest.fixture()
def url_set_config(base_url):
    return f"{base_url}/{OPTIONS_API_ENDPOINT}"


@pytest.mark.parametrize("data", get_all_args())
@pytest.mark.skip()
def test_image_ssim(base_url, data):
    print(f"testing: {get_data_summary(data)}")
    endpoint = TXT2IMG_API_ENDPOINT if is_txt2img(data) else IMG2IMG_API_ENDPOINT
    url = f"{base_url}/{endpoint}"
    generated_image = get_image_array_from_response(post_request_and_check(url, data))
    target_image_path = get_target_image_filename(data)
    directory, filename = os.path.split(target_image_path)
    target_image = np.array(Image.open(target_image_path))
    ssim_value = cal_ssim(generated_image, target_image)
    assert ssim_value > 0.98


def test_onediff_save_graph(url_txt2img):
    script_args = {
        "script_args": [
            False,  # quantization
            None,  # graph_checkpoint
            SAVED_GRAPH_NAME,  # saved_graph_name
        ]
    }
    data = {**get_base_args(), **script_args}
    post_request_and_check(url_txt2img, data)


def test_onediff_load_graph(url_txt2img):
    script_args = {
        "script_args": [
            False,  # quantization
            SAVED_GRAPH_NAME,  # graph_checkpoint
            "",  # saved_graph_name
        ]
    }
    data = {**get_base_args(), **script_args}
    post_request_and_check(url_txt2img, data)
