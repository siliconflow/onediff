import json
import threading
import time
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager

import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)

from .log_utils import logger

__all__ = ["comfy_client_context", "ComfyGraph", "ComfyClient"]


class ComfyGraph:
    def __init__(
        self,
        graph: dict,
        sampler_nodes: list[str],
    ):
        self.graph = graph
        self.sampler_nodes = sampler_nodes

    def set_prompt(self, prompt, negative_prompt=None):
        # Sets the prompt for the sampler nodes (eg. base and refiner)
        for node in self.sampler_nodes:
            prompt_node = self.graph[node]["inputs"]["positive"][0]
            self.graph[prompt_node]["inputs"]["text"] = prompt
            if negative_prompt:
                negative_prompt_node = self.graph[node]["inputs"]["negative"][0]
                self.graph[negative_prompt_node]["inputs"]["text"] = negative_prompt

    def set_sampler_name(
        self,
        sampler_name: str,
    ):
        # sets the sampler name for the sampler nodes (eg. base and refiner)
        for node in self.sampler_nodes:
            self.graph[node]["inputs"]["sampler_name"] = sampler_name

    def set_scheduler(self, scheduler: str):
        # sets the sampler name for the sampler nodes (eg. base and refiner)
        for node in self.sampler_nodes:
            self.graph[node]["inputs"]["scheduler"] = scheduler

    def set_filename_prefix(self, prefix: str):
        # sets the filename prefix for the save nodes
        for node in self.graph:
            if self.graph[node]["class_type"] == "SaveImage":
                self.graph[node]["inputs"]["filename_prefix"] = prefix

    def set_image_size(self, height: int, width: int, batch_size: int = 1):
        for node in self.sampler_nodes:
            size_node = self.graph[node]["inputs"]["latent_image"][0]
            self.graph[size_node]["inputs"]["height"] = height
            self.graph[size_node]["inputs"]["width"] = width
            self.graph[size_node]["inputs"]["batch_size"] = batch_size


@contextmanager
def comfy_client_context(
    listen="127.0.0.1",
    port=30000,
    client_id=str(uuid.uuid4()),
    n_tries=5,
    *args,
    **kwargs,
):
    client = ComfyClient()
    try:
        client.connect(
            listen=listen,
            port=port,
            client_id=client_id,
            n_tries=n_tries,
            *args,
            **kwargs,
        )
        yield client
    except Exception as e:
        logger.error(f"Failed to connect to ComfyClient: {e}")
        raise
    finally:
        logger.info("Closing connection to ComfyClient")
        client.close()


class ComfyClient:
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
                logger.info(f"Connected to server: {self.server_address}")
                threading.Thread(
                    target=self.fetch_system_stats_periodically, daemon=True
                ).start()
                break
        if not self.ws:
            raise RuntimeError(f"Could not connect to server: {self.server_address}")

    def fetch_system_stats_periodically(self):
        max_vram_used_gb = 0
        last_print_time = time.time()
        BYTES_TO_GB = 1024 * 1024 * 1024
        poll_interval = 0.5  # sec
        while True:
            try:
                stats = self.get_system_stats()
                for device in stats["devices"]:
                    vram_total = device["vram_total"]
                    vram_free = device["vram_free"]
                    vram_used = vram_total - vram_free

                    vram_used_gb = vram_used / BYTES_TO_GB

                    if vram_used_gb > max_vram_used_gb:
                        max_vram_used_gb = vram_used_gb

                current_time = time.time()
                if current_time - last_print_time >= 2:
                    logger.info(
                        f"Current VRAM used: {vram_used_gb:.2f} GB\tMaximum VRAM used: {max_vram_used_gb:.2f} GB"
                    )
                    last_print_time = current_time

                time.sleep(poll_interval)
            except Exception as e:
                print(f"{e=}")
                break

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

    def get_system_stats(self):
        with urllib.request.urlopen(
            f"http://{self.server_address}/system_stats"
        ) as response:
            return json.loads(response.read())

    def get_images(self, graph, save=True):
        prompt = graph.graph if isinstance(graph, ComfyGraph) else graph

        if not save:
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
                        break
            else:
                continue

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
