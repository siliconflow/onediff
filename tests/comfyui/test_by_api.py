import argparse
import json
import os

import requests
from PIL import Image

__all__ = ["load_workflow_from_file", "queue_prompt"]

prompt_text = ""


def extract_metadata_from_png(png_file_path):
    img = Image.open(png_file_path)
    metadata = img.info
    img.close()
    return metadata


def load_workflow_from_file(filename) -> str:
    _, extension = os.path.splitext(filename)
    if extension.endswith("json"):
        with open(filename, "r") as f:
            c = f.read()
            return c
    elif extension.endswith("png"):
        metadata = extract_metadata_from_png(filename)
        return metadata["api"]
    else:
        raise NotImplementedError("Only json or png workflow file supported yet")


def parse_args():
    parser = argparse.ArgumentParser(description="Launch ComfyUI workflow by API")
    parser.add_argument(
        "-w",
        "--workflow",
        type=str,
        help="The workflow filename which can be either json or png.",
        required=True,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="ComfyUI server host, default is 127.0.0.1",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8188,
        help="CofyUI server port, default is 8188",
    )

    args = parser.parse_args()
    return args


def queue_prompt(prompt, host, port):
    url = f"http://{host}:{port}/prompt"
    payload = {"prompt": prompt}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        print("Prompt queued successfully.")
        result = json.loads(response.text)
        print(result)
        return result
    else:
        print(
            f"Failed to queue prompt, error code: {response.status_code}, {response.text}"
        )
    return None


if __name__ == "__main__":
    args = parse_args()
    r = load_workflow_from_file(args.workflow)
    prompt = json.loads(r)
    r = queue_prompt(prompt, args.host, args.port)
