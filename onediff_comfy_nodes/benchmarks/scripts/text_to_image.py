import argparse
import sys
import time
from io import BytesIO

import numpy
from PIL import Image

sys.path.append("./src")
from core import ComfyClient, logger, out_dir
from input_registration import get_input_constructor

DEFAULT_TIMEOUT = 200
DEFAULT_HOST = "127.0.0.1"
DEFAULT_COMFY_PORT = "8188"


def parse_args():
    parser = argparse.ArgumentParser(description="Test ComfyUI workflow by Selenium.")
    parser.add_argument(
        "-w", "--workflow", type=str, required=True, help="Workflow file"
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Timeout for Selenium requests",
    )
    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST, help="Selenium service host"
    )
    parser.add_argument(
        "--comfy-port",
        type=str,
        default=DEFAULT_COMFY_PORT,
        help="ComfyUI service port",
    )
    parser.add_argument(
        "--comfy-pid", type=int, default=None, help="ComfyUI process ID (optional)"
    )
    return parser.parse_args()


def run_text_to_image_benchmark(
    workflow="resources/example_workflow_api.json",
    comfy_port=DEFAULT_COMFY_PORT,
    comfy_pid=None,
):
    client = ComfyClient()
    client.connect(port=comfy_port)
    client.comfy_pid = comfy_pid

    for test_graph in get_input_constructor(workflow):
        start_time = time.time()
        images = client.get_images(test_graph)
        assert len(images) != 0, "No images generated"
        # assert all images are not blank
        duration = time.time() - start_time
        logger.info(f"Benchmark duration: {duration} seconds")
        for images_output in images.values():
            for image_data in images_output:
                pil_image = Image.open(BytesIO(image_data))
                assert numpy.array(pil_image).any() != 0, "Image is blank"

    client.close()


if __name__ == "__main__":
    args = parse_args()
    separator = "=" * 20
    print(f"{separator}\nResult: {out_dir}\n{separator}")

    run_text_to_image_benchmark(args.workflow, args.comfy_port, args.comfy_pid)
