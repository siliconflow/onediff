import os
import json

import requests
from PIL import Image

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
        "-h",
        "--host",
        type=str,
        default="127.0.0.1",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8188,
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
    else:
        print("Failed to queue prompt.")
    print(response.content)


if __name__ == "__main__":
    args = parse_args()
    r = load_workflow_from_file(args.workflow)
    prompt = json.loads(r)
    queue_prompt(prompt, args.host, args.port)
