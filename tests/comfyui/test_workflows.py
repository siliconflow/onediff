import os
import sys
import json
import time

import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from comfy_queue_prompt import *


def is_finished(prompt_ids, history):
    def get_output_info(history_info):
        outputs_dict = history_info["outputs"]

        for _, v in outputs_dict.items():
            images_list = v["images"]
            for image in images_list:
                return image["filename"]

    keys_to_remove = []
    for k in prompt_ids.keys():
        if k in history.keys():
            print(f"{k} has finished, outputfile: {get_output_info(history[k])}")
            keys_to_remove.append(k)
    for key in keys_to_remove:
        prompt_ids.pop(key)
    if len(prompt_ids) == 0:
        return True
    return False


if __name__ == "__main__":
    PORT = 8855
    files = ["txt2img.png", "int8txt2img.png"]
    workflows = [os.path.join(current_dir, "workflows", file) for file in files]

    prompt_ids = {}
    for workflow in workflows:
        r = load_workflow_from_file(workflow)
        prompt = json.loads(r)
        r = queue_prompt(prompt, "127.0.0.1", PORT)
        prompt_ids[r["prompt_id"]] = r

    time.sleep(60 * 1)
    url = f"http://127.0.0.1:{PORT}/history"

    start_time = time.time()
    while True:
        response = requests.get(url)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time > 60 * 15:
            print(f"time out: {elapsed_time} seconds elapsed")
            break

        if response.status_code == 200:
            data = response.json()
            if is_finished(prompt_ids, data):
                print("All prompts finished")
                break
        else:
            print("Failed to fetch data.")
            exit(1)

        time.sleep(5)
