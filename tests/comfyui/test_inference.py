"""
ci mem limit: 12288MiB

Run the test script:
python3 tests/comfyui/test_inference.py -w  /share_nfs/hf_models/comfyui_resources/workflow_apis/comfyui_basic.json
python tests/comfyui/test_inference.py -w  /share_nfs/hf_models/comfyui_resources/workflow_apis/comfyui_lora.json
python tests/comfyui/test_inference.py -w  /share_nfs/hf_models/comfyui_resources/workflow_apis/comfyui_controlnet-lora.json
"""
from copy import deepcopy
from typing import Tuple, Union
import argparse
import numpy as np
import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import signal
import time
from skimage.metrics import structural_similarity as ssim


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Execution timed out")


signal.signal(signal.SIGALRM, timeout_handler)


def parse_args():
    parser = argparse.ArgumentParser()
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
        default="100",
    )
    # ssim_threshold
    parser.add_argument(
        "-s",
        "--ssim_threshold",
        type=float,
        default="0.8",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=3000,
        help="Port to connect to",
    )
    parser.add_argument("-l", "--listen", type=str, default="127.0.0.1")
    return parser.parse_args()


def ssim_score(img0: np.ndarray, img1: np.ndarray) -> Tuple[float, np.ndarray]:
    score, diff = ssim(img0, img1, channel_axis=-1, full=True)
    # rescale the difference image to 0-255 range
    diff = (diff * 255).astype("uint8")
    return score, diff


def To_CheckpointLoaderSimple(node):
    node["class_type"] = "CheckpointLoaderSimple"
    node["inputs"] = {"ckpt_name": node["inputs"]["ckpt_name"]}
    node["_meta"] = {"title": "Load Checkpoint"}
    return node


def To_ControlNetLoader(node):
    node["class_type"] = "ControlNetLoader"
    node["inputs"] = {"control_net_name": node["inputs"]["control_net_name"]}
    node["_meta"] = {"title": "Load ControlNet Model"}
    return node


ONEDIFF_TO_TORCH = {
    "OneDiffCheckpointLoaderSimple": To_CheckpointLoaderSimple,
    "OneDiffControlNetLoader": To_ControlNetLoader,
}


class ComfyClient:
    # From examples/websockets_api_example.py

    def connect(
        self,
        listen: str = "127.0.0.1",
        port: Union[str, int] = 30000,
        client_id: str = str(uuid.uuid4()),
    ):
        self.client_id = client_id
        self.server_address = f"{listen}:{port}"
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(self.server_address, self.client_id))
        self.ws = ws

    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(
            "http://{}/prompt".format(self.server_address), data=data
        )
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(
            "http://{}/view?{}".format(self.server_address, url_values)
        ) as response:
            return response.read()

    def get_history(self, prompt_id):
        with urllib.request.urlopen(
            "http://{}/history/{}".format(self.server_address, prompt_id)
        ) as response:
            return json.loads(response.read())

    def get_images(self, graph, save=True):
        prompt = graph
        if not save:
            # Replace save nodes with preview nodes
            prompt_str = json.dumps(prompt)
            prompt_str = prompt_str.replace("SaveImage", "PreviewImage")
            prompt = json.loads(prompt_str)

        prompt_id = self.queue_prompt(prompt)["prompt_id"]
        output_images = {}
        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message["type"] == "executing":
                    data = message["data"]
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        break  # Execution is done
            else:
                continue  # previews are binary data

        history = self.get_history(prompt_id)[prompt_id]
        for o in history["outputs"]:
            for node_id in history["outputs"]:
                node_output = history["outputs"][node_id]
                if "images" in node_output:
                    images_output = []
                    for image in node_output["images"]:
                        image_data = self.get_image(
                            image["filename"], image["subfolder"], image["type"]
                        )
                        images_output.append(image_data)
                output_images[node_id] = images_output

        return output_images


class ComfyGraph:
    def __init__(self, graph):
        self.graph = graph

    def set_random_seed(self, seed):
        for k, node in self.graph.items():
            if node["class_type"] == "KSampler":
                node["inputs"]["seed"] = seed

    def set_bwh(self, batch_size, width, height):
        """set_batch_width_height"""
        for k, node in self.graph.items():
            if node["class_type"] == "EmptyLatentImage":
                node["inputs"]["batch_size"] = batch_size
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height

    @property
    def torch_graph(self):
        return self.to_torch_graph()

    def to_torch_graph(self):
        torch_graph = {}
        original_graph = deepcopy(self.graph)

        for k, node in original_graph.items():
            conversion_func = ONEDIFF_TO_TORCH.get(node["class_type"], lambda x: x)
            torch_graph[k] = conversion_func(node)

        return torch_graph


def validate_inference_consistency(client, comfy_graph, ssim_threshold=0.85):
    print("Validating Inference Consistency")
    start_time = time.time()
    reference_images = client.get_images(comfy_graph.torch_graph)
    target_images = client.get_images(comfy_graph.graph)

    for ref_id, target_id in zip(reference_images, target_images):
        for ref_image_data, target_image_data in zip(
            reference_images[ref_id], target_images[target_id]
        ):
            from PIL import Image
            import io

            ref_image = Image.open(io.BytesIO(ref_image_data))
            ref_image = np.array(ref_image)

            target_image = Image.open(io.BytesIO(target_image_data))
            target_image = np.array(target_image)

            score, diff = ssim_score(ref_image, target_image)
            assert score > ssim_threshold, f"SSIM score: {score}, diff: {diff}"
            print(f"SSIM score: {score}")

    print("Inference Consistency Validation Done", time.time() - start_time, "seconds")


def run_inference_tests(client, comfy_graph):
    # Test inference
    print("Running Inference Tests")
    start_time = time.time()
    for b in [1, 2]:
        for h in [256, 64]:
            for w in [256, 64]:
                comfy_graph.set_bwh(b, w, h)
                client.get_images(comfy_graph.graph)

    print("Inference Tests Done", time.time() - start_time, "seconds")
    


def get_comfy_graph(file_path):
    assert file_path.endswith(".json")
    with open(file_path, "r") as f:
        comfy_graph = json.load(f)
    return comfy_graph


if __name__ == "__main__":
    args = parse_args()
    client = ComfyClient()
    client.connect(listen=args.listen, port=args.port)
    graph = get_comfy_graph(args.workflow)
    comfy_graph = ComfyGraph(graph)
    signal.alarm(args.timeout)

    try:
        validate_inference_consistency(
            client, comfy_graph, ssim_threshold=args.ssim_threshold
        )
        run_inference_tests(client, comfy_graph)
    except TimeoutException as e:
        raise RuntimeError(f"Timeout after {args.timeout} seconds") from e
    finally:
        signal.alarm(0)
