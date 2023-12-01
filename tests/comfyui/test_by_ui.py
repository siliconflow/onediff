import argparse
import os
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


def parse_args():
    parser = argparse.ArgumentParser(description="Test ComfyUI workflow by Selenium.")
    parser.add_argument(
        "-w",
        "--workflow",
        type=str,
        default="Workflow file",
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
    args = parser.parse_args()
    return args


args = parse_args()


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


def launch_prompt(driver):
    try:
        driver.get("http://127.0.0.1:8855")
        time.sleep(0.1)
        start_time = time.time()

        wait_until_app_ready(driver)

        clear_curernt_workflow(driver)

        load_workflow_graph(driver, read_workflow_json(args.workflow))
        print(f"{args.workflow} loaded, and launch the queue prompt...")

        check_graph_node_types(driver)
        print(f"{args.workflow} workflow checked")

        launch_and_wait(driver, timeout=args.timeout)
        print(
            f"{args.workflow} has finished, time elapsed: {time.time() - start_time:.1f}"
        )
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
    driver = webdriver.Remote(
        command_executor=f"http://{args.host}:{args.port}/wd/hub", options=options
    )
    launch_prompt(driver)
