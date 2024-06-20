import argparse
import os
import sys
import time
from io import BytesIO

import numpy
from PIL import Image

sys.path.append("./src")
from core.log_utils import setup_logging
from core.service_client import comfy_client_context
from input_registration import get_input_constructor

DEFAULT_HOST = "127.0.0.1"
DEFAULT_COMFY_PORT = "8188"


def parse_args():
    parser = argparse.ArgumentParser(description="Test ComfyUI workflow")
    parser.add_argument(
        "-w", "--workflow", type=str, required=True, help="Workflow file"
    )
    
    parser.add_argument(
        "--listen", type=str, default=DEFAULT_HOST, help="service listen"
    )
    parser.add_argument(
        "--comfy-port",
        type=str,
        default=DEFAULT_COMFY_PORT,
        help="ComfyUI service port",
    )
    parser.add_argument(
        "--output-images", action="store_true", help="Enable output of images."
    )
    return parser.parse_args()


def run_workflow(
    workflow="resources/example_workflow_api.json",
    comfy_port=DEFAULT_COMFY_PORT,
    output_images=True,
):
    logger, result_dir = setup_logging(exp_name="exp")
    print(f"====\n {result_dir=}\n====")

    with comfy_client_context(port=comfy_port) as client:
        logger.info(f"Test {workflow}")
        for i, comfy_graph in enumerate(get_input_constructor(workflow)):
            start_time = time.time()
            images = client.get_images(comfy_graph.graph)
            assert len(images) != 0, "No images generated"
            # assert all images are not blank
            duration = time.time() - start_time
            logger.info(f"E2E time: {duration} seconds")
            for images_output in images.values():
                for image_data in images_output:
                    pil_image = Image.open(BytesIO(image_data))
                    assert numpy.array(pil_image).any() != 0, "Image is blank"
                    if output_images:
                        out_img_path = os.path.join(result_dir, "imgs")
                        os.makedirs(out_img_path, exist_ok=True)
                        image_path = os.path.join(out_img_path, f"image_{i}.png")
                        logger.info(f"Saved image to: {image_path}")
                        pil_image.save(image_path)

if __name__ == "__main__":
    args = parse_args()
    run_workflow(args.workflow, args.comfy_port, args.output_images)
