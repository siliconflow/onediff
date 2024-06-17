import json
import time
import urllib.parse
import urllib.request
import uuid

import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)

from .log_utils import logger
from .monitor_memory import get_process_memory


class ComfyClient:
    def __init__(self) -> None:
        self.comfy_pid = None
        self.max_mem_usage = -1

    def connect(
        self, listen="127.0.0.1", port=30000, client_id=str(uuid.uuid4()), n_tries=5
    ):
        for i in range(n_tries):
            time.sleep(4)
            self.client_id = client_id
            self.server_address = f"{listen}:{port}"
            try:
                ws = websocket.WebSocket()
                ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
                self.ws = ws
            except ConnectionRefusedError as e:
                print(e)
                print(f"({i+1}/{n_tries}) Retrying...")
            else:
                print("Connected to server: ", self.server_address)
                break
        if not self.ws:
            raise RuntimeError(f"Could not connect to server: {self.server_address}")

    def close(self):
        self.ws.close()

    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(
            f"http://{self.server_address}/view?{url_values}"
        ) as response:
            return response.read()

    def get_history(self, prompt_id):
        with urllib.request.urlopen(
            f"http://{self.server_address}/history/{prompt_id}"
        ) as response:
            return json.loads(response.read())

    def get_images(self, graph, save=True):
        prompt = graph
        if not save:
            prompt_str = json.dumps(prompt)
            prompt_str = prompt_str.replace("SaveImage", "PreviewImage")
            prompt = json.loads(prompt_str)

        prompt_id = self.queue_prompt(prompt)["prompt_id"]
        output_images = {}
        _count = 0
        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message["type"] == "executing":
                    data = message["data"]
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        break
            else:
                continue

            if _count % 100 == 0 and self.comfy_pid is not None:
                mem_usage = get_process_memory(
                    self.comfy_pid
                )  # Running once takes about 0.02 seconds
                if mem_usage is None:
                    logger.warning(f"{self.comfy_pid=} error")
                    self.comfy_pid = None
                else:
                    if mem_usage > self.max_mem_usage:
                        self.max_mem_usage = mem_usage
                    logger.info(
                        f"Process {self.comfy_pid} is using {mem_usage} MB of GPU memory"
                    )
            _count += 1

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
