import argparse
import os
import sys
import time
from io import BytesIO
from typing import List, Union

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

sys.path.append("./src")

from core.log_utils import setup_logging
from core.service_client import comfy_client_context
from input_registration import dispatch_generator

DEFAULT_HOST = "127.0.0.1"
DEFAULT_COMFY_PORT = "8188"


def parse_args():
    parser = argparse.ArgumentParser(description="Test ComfyUI workflow")
    parser.add_argument(
        "-w", "--workflow", type=str, nargs="+", required=True, help="Workflow file(s)"
    )
    parser.add_argument(
        "--listen", type=str, default=DEFAULT_HOST, help="Service listen address"
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
    parser.add_argument(
        "--baseline-dir", type=str, help="Directory for baseline output."
    )
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.5,
        help="SSIM threshold for image comparison.",
    )
    return parser.parse_args()


def save_image(image_data: bytes, output_dir: str, image_name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, image_name)
    Image.open(BytesIO(image_data)).save(image_path)
    return image_path


def calculate_ssim(image1: Image.Image, image2: Image.Image) -> float:
    image1_np = np.array(image1.convert("RGB"))
    image2_np = np.array(image2.convert("RGB"))
    assert image1_np.shape == image2_np.shape, "Images must have the same dimensions"
    return ssim(image1_np, image2_np, channel_axis=2)


class WorkflowProcessor:
    def __init__(
        self,
        output_images: bool,
        output_dir: str,
        baseline_dir: str,
        logger,
        ssim_threshold,
    ):
        self.output_images = output_images
        self.output_dir = output_dir
        self.baseline_dir = baseline_dir
        self.logger = logger
        self.ssim_threshold = ssim_threshold

    def process_image(self, image_data: bytes, index: int) -> None:
        pil_image = Image.open(BytesIO(image_data))
        self.logger.info(
            f"Image Size - Height: {pil_image.height}px, Width: {pil_image.width}px"
        )
        assert np.array(pil_image).any() != 0, "Image is blank"

        if self.output_images:
            image_path = save_image(image_data, self.output_dir, f"image_{index}.png")
            self.logger.info(f"Saved image to: {image_path}")

        if self.baseline_dir:
            baseline_image_path = os.path.join(self.baseline_dir, f"image_{index}.png")
            baseline_image = Image.open(baseline_image_path)
            ssim_value = calculate_ssim(pil_image, baseline_image)
            self.logger.info(f"SSIM value with baseline: {ssim_value}")
            assert ssim_value > self.ssim_threshold


def run_workflow(
    workflow: Union[str, List[str]],
    comfy_port: str = DEFAULT_COMFY_PORT,
    output_images: bool = True,
    baseline_dir: str = None,
    ssim_threshold: float = 0.5,
) -> None:
    logger, result_dir = setup_logging(exp_name="exp")
    logger.info(f"Result directory: {result_dir}")

    processor = WorkflowProcessor(
        output_images,
        os.path.join(result_dir, "imgs"),
        baseline_dir,
        logger,
        ssim_threshold,
    )

    with comfy_client_context(port=comfy_port) as client:
        logger.info(f"Testing workflows: {workflow}")
        for i, comfy_graph in enumerate(dispatch_generator(workflow)):
            start_time = time.time()
            images = client.get_images(comfy_graph.graph)
            end_time = time.time()

            if not images:
                logger.error("No images generated")
                raise ValueError("No images generated")
            if len(images) != 1:
                logger.error(
                    f"Expected 1 image, but got {len(images)} images. Batch Size == 1"
                )
                raise ValueError(
                    f"Expected 1 image, but got {len(images)} images. Batch Size == 1"
                )

            for images_output in images.values():
                for image_data in images_output:
                    processor.process_image(image_data, i)

            e2e_time = end_time - start_time
            logger.info(f"Workflow {i} E2E:  {e2e_time:.2f} seconds")
            # if i>0: # TODO refine
            #     assert e2e_time < 10 # sec


if __name__ == "__main__":
    args = parse_args()
    run_workflow(
        args.workflow,
        args.comfy_port,
        args.output_images,
        args.baseline_dir,
        args.ssim_threshold,
    )
