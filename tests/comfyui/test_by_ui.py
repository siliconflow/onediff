"""
Before running this script, you need to start the Selenium and ComfyUI services.
You can start their containers using Docker Compose.

Please set the following environment variables (whose values are for reference only):

export ACR_ORG=registry.cn-beijing.aliyuncs.com/oneflow
export MATRIX_IMAGE=onediff-pro:cu121
export COMFYUI_SRC_DIR=ComfyUI
export SELENIUM_IMAGE=standalone-chrome:119.0-chromedriver-119.0-grid-4.15.0-20231129
export SELENIUM_CONTAINER_NAME=selenium-test
export COMFYUI_PORT=8855
export CONTAINER_NAME=onediff-test
export SDXL_BASE=/share_nfs/hf_models/sd_xl_base_1.0.safetensors
export UNET_INT8=/share_nfs/hf_models/unet_int8
export CONTROL_LORA_OPENPOSEXL2_RANK256=/share_nfs/hf_models/controlnet/control-lora-openposeXL2-rank256.safetensors

And then:

**Note**:
  It is advisable to execute the following commands in the 'diffusers' directory
unless you are fully aware of the implications of executing them in a different
directory.

git clone https://github.com/comfyanonymous/ComfyUI.git

docker compose -f tests/comfy-docker-compose.yml up -d
docker exec $CONTAINER_NAME python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
docker exec $CONTAINER_NAME python3 -m pip install -r tests/comfyui/requirements.txt --user
docker exec $CONTAINER_NAME python3 -m pip install -r /app/ComfyUI/requirements.txt --user
docker exec -it $CONTAINER_NAME python3 /app/ComfyUI/main.py --cuda-device 0

Run the test script:

python tests/comfyui/test_by_ui.py --comfy_port 8188 --workflow tests/comfyui/workflows/sdxl-unet-speedup-graph-saver.json

If you need to shutdown the test containers, run:

docker compose -f tests/comfy-docker-compose.yml down
"""
import argparse
import os
import time

from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def parse_args():
    parser = argparse.ArgumentParser(description="Test ComfyUI workflow by Selenium.")
    parser.add_argument(
        "-w",
        "--workflow",
        type=str,
        required=True,
        help="Workflow file",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default="200",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="The selenium service host",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="4444",
        help="The selenium service port",
    )
    parser.add_argument(
        "--comfy_port",
        type=str,
        default="8188",
        help="The ComfyUI service port",
    )
    args = parser.parse_args()
    return args


args = parse_args()


def extract_metadata_from_png(png_file_path):
    img = Image.open(png_file_path)
    metadata = img.info
    img.close()
    return metadata


def read_workflow_json(filename) -> str:
    _, extension = os.path.splitext(filename)
    if extension.endswith("json"):
        with open(filename, "r") as f:
            c = f.read()
            return c
    elif extension.endswith("png"):
        metadata = extract_metadata_from_png(filename)
        return metadata["workflow"]
    else:
        raise NotImplementedError("Only json or png workflow file supported yet")


def load_workflow_graph(driver, workflow: str):
    driver.execute_script(f"window.app.loadGraphData({workflow})")


def click_queue_prompt_button(driver):
    wait = WebDriverWait(driver, 1)
    queue_button = wait.until(EC.presence_of_element_located((By.ID, "queue-button")))
    queue_button.click()


def clear_curernt_workflow(driver):
    driver.execute_script("window.app.graph.clear()")


def wait_until_queue_finished(driver, timeout=100):
    time.sleep(0.3)
    wait = WebDriverWait(driver, timeout)
    element = wait.until(
        EC.presence_of_element_located(
            (By.XPATH, f'//*[contains(text(), "Queue size: 0")]')
        )
    )


def wait_until_app_ready(driver):
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.ID, "comfy-clear-button"))
    )
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.ID, "graph-canvas"))
    )


def launch_and_wait(driver, *, timeout=100):
    click_queue_prompt_button(driver)
    wait_until_queue_finished(driver, timeout)


def check_graph_node_types(driver):
    try:
        driver.execute_script("window.graph.checkNodeTypes()")
    except Exception as e:
        raise Exception("Error: Workflow nodes checking failed, likely missing nodes")


def check_error_occurs(driver):
    elements = driver.find_elements(By.CLASS_NAME, "comfy-modal-content")

    desired_element = None
    for element in elements:
        element_text = element.text
        if "Error occurred when" in element_text:
            print(element.text)
            raise Exception(f"{element.text}")


def launch_prompt(driver):
    try:
        print(f"connect to ComfyUI: 127.0.0.1:{args.comfy_port}...")
        driver.get(f"http://127.0.0.1:{args.comfy_port}")
        print(f"ComfyUI connected")
        time.sleep(0.1)
        start_time = time.time()

        wait_until_app_ready(driver)

        print("clear the workflow...")
        clear_curernt_workflow(driver)
        print("workflow cleard")

        print("load the target workflow...")
        load_workflow_graph(driver, read_workflow_json(args.workflow))
        print(f"{args.workflow} loaded")

        print("check the nodes type of workflow...")
        check_graph_node_types(driver)
        print(f"{args.workflow} workflow checked")

        print(f"launch the queue prompt (timeout: {args.timeout}s) ...")
        launch_and_wait(driver, timeout=args.timeout)

        duration = time.time() - start_time
        print(f"{args.workflow} has finished, time elapsed: {duration:.1f}")

        if duration < 2:
            raise ValueError(
                "Execution duration is too short, possible error in workflow execution"
            )

        print(f"check if error occurs...")
        check_error_occurs(driver)
        print(f"no error occurs when executing workflow")
    except TimeoutException:
        print("Time out")
        exit(1)
    except Exception as e:
        print(type(e))
        print(e)
        print("exit with error: 1")
        exit(1)
    finally:
        driver.quit()


if __name__ == "__main__":
    options = webdriver.ChromeOptions()
    print(f"connect remote Selenium: http://{args.host}:{args.port}/wd/hub ...")
    driver = webdriver.Remote(
        command_executor=f"http://{args.host}:{args.port}/wd/hub", options=options
    )
    print(f"Selenium connected")
    launch_prompt(driver)
