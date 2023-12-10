import folder_paths
import random
from PIL import Image
import os
import numpy as np
import torch
import subprocess

try:
    from skimage.metrics import structural_similarity
except ImportError as e:
    subprocess.check_call(["pip", "install", "scikit-image"])
    from skimage.metrics import structural_similarity


class CompareModel:
    r"""
    This node is used to compare the weight values corresponding to each same key in PyTorch and Oneflow Graph,
    and to verify the correctness of Oneflow Graph compilation, especially for models with LoRA.

    To use it, you MUST load two same checkpoints (one for PyTorch and another for OneFlow Graph)
    and connect them to the two inputs of the node, then set `check` to enable.

    If there is too much difference between their weights, the node will raise RuntimeError.

    Note:
        The first time you run workflow with OneFlow Graph, the node cannot capture the weights of OneFlow Graph without compiling,
        so you need to run workflow at least twice to ensure that this node works properly.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "torch_model": ("MODEL",),
                "oneflow_model": ("MODEL",),
                "check": (["enable", "disable"],),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "compare"
    CATEGORY = "OneDiff"

    def compare(self, torch_model, oneflow_model, check):
        if check == "disable":
            return {}

        import numpy as np
        import oneflow as flow
        from onediff.infer_compiler.transform.builtin_transform import torch2oflow

        torch_model.patch_model("cuda")
        oneflow_model.patch_model("cuda")

        torch_unet = torch_model.model.diffusion_model
        oflow_unet = oneflow_model.model.diffusion_model

        if oflow_unet._deployable_module_model._oneflow_module is None:
            torch_model.unpatch_model("cuda")
            oneflow_model.unpatch_model("cuda")
            print(
                "\033[91m",
                f"[Compare Model Diff] Cannot get compiled oneflow graph, maybe you need to build it"
                + "\033[0m",
            )
            return {}

        cnt = 0
        for key, _ in oflow_unet.named_parameters():
            key = key.removeprefix("_deployable_module_model._torch_module.")
            torch_value = torch_unet.get_parameter(key).cuda()
            oflow_value = (
                oflow_unet._deployable_module_model._oneflow_module.get_parameter(
                    key
                ).cuda()
            )

            if not flow.allclose(torch2oflow(torch_value), oflow_value, 1e-4, 1e-4):
                print(
                    "\033[91m",
                    f"value of key [{key}] is different, max diff is {(torch2oflow(torch_value) - oflow_value).float().abs().max().item()}"
                    + "\033[0m",
                )
                cnt += 1

            if cnt >= 10:
                torch_model.unpatch_model("cuda")
                oneflow_model.unpatch_model("cuda")
                raise RuntimeError(
                    "Too much lora weight diff between torch and oneflow, omit..."
                )

        torch_model.unpatch_model("cuda")
        oneflow_model.unpatch_model("cuda")
        return {}


class ShowImageDiff:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5)
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images1": ("IMAGE",),
                "images2": ("IMAGE",),
                "rtol": ("STRING", {"default": "1e-4"}),
                "atol": ("STRING", {"default": "1e-4"}),
                "raise_if_diff": (["enable", "disable"],),
                "image_id": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "OneDiff"

    def save_images(
        self,
        images1,
        images2,
        rtol: str,
        atol: str,
        raise_if_diff: str,
        image_id="ComfyUI",
        prompt=None,
        extra_pnginfo=None,
    ):
        assert len(images1) == len(images2)
        filename_prefix = image_id[:]
        filename_prefix += self.prefix_append
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            filename_prefix,
        ) = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            images1[0].shape[1],
            images1[0].shape[0],
        )
        results = list()
        for image1, image2 in zip(images1, images2):
            # image diff
            image = image1 - image2

            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            file = f"{filename}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=4,
            )
            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

            max_diff = image.abs().max().item()

            img1 = self.image_to_numpy(image1)
            img2 = self.image_to_numpy(image2)
            ssim = structural_similarity(img1, img2, channel_axis=2)

            print(
                "\033[91m"
                f"[ShowImageDiff {image_id}] Max value of diff is {max_diff}, image simularity is {ssim:.6f}"
                + "\033[0m"
            )

            if raise_if_diff == "enable":
                default_rtol = 1e-4
                default_atol = 1e-4
                try:
                    rtol = float(rtol)
                except ValueError:
                    print(
                        "\033[91m"
                        + f"rtol = {rtol} is not a float, use default value rtol={default_rtol}"
                        + "\033[0m"
                    )
                    rtol = default_rtol
                try:
                    atol = float(atol)
                except ValueError:
                    print(
                        "\033[91m"
                        + f"atol = {atol} is not a float, use default value atol={default_atol}"
                        + "\033[0m"
                    )
                    atol = default_atol

                if not torch.allclose(image1, image2, rtol=rtol, atol=atol):
                    raise RuntimeError(
                        f"The diff between image1 and image2 is beyond the scope of rtol={rtol}, atol={atol}, max diff is {max_diff}"
                    )

        return {"ui": {"images": results}}

    def image_to_numpy(self, image):
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return np.array(img)
