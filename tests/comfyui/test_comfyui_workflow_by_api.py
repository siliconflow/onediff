import os
import json
from urllib import request, parse

from PIL import Image

prompt_text = ""



def extract_metadata_from_png(png_file_path):
        img = Image.open(png_file_path)
        metadata = img.info
        img.close()
        return metadata

def load_workflow_from_file(filename):
    _, extension = os.path.splitext(filename)
    if extension.endswith("json"):
        with open(filename, "r") as f:
            c = f.read()
            prompt = json.loads(c)
            return prompt
    elif extension.endswith("png"):
        metadata = extract_metadata_from_png(filename)
        return metadata['api']
    else:
        raise NotImplementedError("Only json or png workflow file supported yet")

def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://127.0.0.1:8855/prompt", data=data)
    request.urlopen(req)


if __name__ == "__main__":
    r = load_workflow_from_file("/data/home/yaochi/pack-ComfyUI-docker-image/diffusers/tests/comfyui/workflows/processed-lora-lora.png")
    print(r)

    # try:
    #     queue_prompt(prompt)
    # except Exception as e:
    #     print("!!!", e)

