import warnings
from typing import List, Optional

import numpy as np
import PIL.Image
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import deprecate
from PIL import Image


def patch_image_prcessor(processor):
    if type(processor) is VaeImageProcessor:
        processor.postprocess = postprocess.__get__(processor)
        processor.pt_to_numpy = pt_to_numpy.__get__(processor)
        processor.pt_to_pil = pt_to_pil.__get__(processor)
    else:
        warnings.warn(
            f"Image processor {type(processor)} is not supported for patching"
        )


def postprocess(
    self,
    image: torch.FloatTensor,
    output_type: str = "pil",
    do_denormalize: Optional[List[bool]] = None,
):
    if not isinstance(image, torch.Tensor):
        raise ValueError(
            f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor"
        )
    if output_type not in ["latent", "pt", "np", "pil"]:
        deprecation_message = (
            f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
            "`pil`, `np`, `pt`, `latent`"
        )
        deprecate(
            "Unsupported output_type", "1.0.0", deprecation_message, standard_warn=False
        )
        output_type = "np"

    if output_type == "latent":
        return image

    if do_denormalize is None:
        do_denormalize = [self.config.do_normalize] * image.shape[0]

    image = torch.stack(
        [
            self.denormalize(image[i]) if do_denormalize[i] else image[i]
            for i in range(image.shape[0])
        ]
    )

    if output_type == "pt":
        return image

    if output_type == "pil":
        return self.pt_to_pil(image)

    image = self.pt_to_numpy(image)

    if output_type == "np":
        return image

    # if output_type == "pil":
    #     return self.numpy_to_pil(image)


@torch.jit.script
def _pt_to_numpy_pre(images):
    return images.permute(0, 2, 3, 1).contiguous().float().cpu()


@staticmethod
def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    # images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    # return images
    return _pt_to_numpy_pre(images).numpy()


def _pt_to_pil_pre(images):
    return (
        images.permute(0, 2, 3, 1)
        .contiguous()
        .float()
        .mul(255)
        .round()
        .to(dtype=torch.uint8)
        .cpu()
    )


@staticmethod
def pt_to_pil(images: np.ndarray) -> PIL.Image.Image:
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    # images = (images * 255).round().astype("uint8")
    images = _pt_to_pil_pre(images).numpy()
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images
