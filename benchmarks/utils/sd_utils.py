import os
import importlib
import inspect
import argparse
import time
import json
import torch
from PIL import Image, ImageDraw
import numpy as np
from diffusers.utils import load_image
from safetensors.torch import load_file

import oneflow as flow
from onediffx import compile_pipe


def load_sd_pipe(
    pipeline_cls,
    model_name,
    dtype=torch.float16,
    variant=None,
    custom_pipeline=None,
    scheduler=None,
    lora=None,
    controlnet=None,
    device=None,
):
    extra_kwargs = {}
    if custom_pipeline is not None:
        extra_kwargs["custom_pipeline"] = custom_pipeline
    if variant is not None:
        extra_kwargs["variant"] = variant
    if controlnet is not None:
        from diffusers import ControlNetModel

        controlnet = ControlNetModel.from_pretrained(
            controlnet,
            torch_dtype=dtype,
        )
        extra_kwargs["controlnet"] = controlnet
    if os.path.exists(os.path.join(model_name, "calibrate_info.txt")):
        from onediff.quantization import QuantPipeline

        pipe = QuantPipeline.from_quantized(
            pipeline_cls, model_name, torch_dtype=dtype, **extra_kwargs
        )
        print("Quantized model loaded  successfully")
    else:
        pipe = pipeline_cls.from_pretrained(
            model_name, torch_dtype=dtype, **extra_kwargs
        )
    if scheduler is not None:
        scheduler_cls = getattr(importlib.import_module("diffusers"), scheduler)
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    if lora is not None:
        pipe.load_lora_weights(lora)
        pipe.fuse_lora()
    pipe.safety_checker = None
    # TODO add npu device
    pipe.to(device)
    return pipe


def load_sd_light_pipe(
    pipeline_cls,
    model_name,
    dtype=torch.float16,
    variant=None,
    custom_pipeline=None,
    scheduler=None,
    lora=None,
    controlnet=None,
    device=None,
    repo_name=None,
    cpkt_name=None,
):

    extra_kwargs = {}
    if custom_pipeline is not None:
        extra_kwargs["custom_pipeline"] = custom_pipeline
    if variant is not None:
        extra_kwargs["variant"] = variant
    if controlnet is not None:
        from diffusers import ControlNetModel

        controlnet = ControlNetModel.from_pretrained(
            controlnet,
            torch_dtype=dtype,
        )
        extra_kwargs["controlnet"] = controlnet
    if os.path.exists(os.path.join(model_name, "calibrate_info.txt")):
        from onediff.quantization import QuantPipeline

        raise TypeError("Quantizatble SDXL-LIGHT is not supported!")
        # pipe = QuantPipeline.from_quantized(
        #     pipeline_cls, model_name, torch_dtype=torch.float16, **extra_kwargs
        # )
    else:
        is_lora_cpkt = "lora" in cpkt_name

        pipe = pipeline_cls.from_pretrained(
            model_name, torch_dtype=dtype, **extra_kwargs
        )
        if is_lora_cpkt:
            pipe = pipeline_cls.from_pretrained(
                model_name, torch_dtype=dtype, **extra_kwargs
            ).to(device)
            if os.path.isfile(os.path.join(repo_name, cpkt_name)):
                pipe.load_lora_weights(os.path.join(repo_name, cpkt_name))
            else:
                pipe.load_lora_weights(hf_hub_download(repo_name, cpkt_name))
            pipe.fuse_lora()
        else:
            from diffusers import UNet2DConditionModel

            unet = UNet2DConditionModel.from_config(model_name, subfolder="unet").to(
                device, dtype
            )

            if os.path.isfile(os.path.join(repo_name, cpkt_name)):

                unet.load_state_dict(
                    load_file(os.path.join(repo_name, cpkt_name), device="cuda")
                )
            else:
                from huggingface_hub import hf_hub_download

                unet.load_state_dict(
                    load_file(hf_hub_download(repo_name, cpkt_name), device="cuda")
                )
            pipe = pipeline_cls.from_pretrained(
                model_name, unet=unet, torch_dtype=dtype, **extra_kwargs
            ).to(device)
    if scheduler is not None:
        scheduler_cls = getattr(importlib.import_module("diffusers"), scheduler)
        pipe.scheduler = scheduler_cls.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )

    pipe.safety_checker = None
    pipe.to(device)
    return pipe


# TODO add npu
def get_device(device: str):
    if device is None:
        raise ValueError("Please specify the device")
    elif device == "cuda":
        return torch.device("cuda")
    else:
        raise ValueError("Unknown device")


def get_input_image(image: str, width: int, height: int):
    image = load_image(image)
    image = image.resize((width, height), Image.LANCZOS)
    return image


def get_control_image(controlnet, control_image: str, width: int, height: int):
    if control_image is None:
        if controlnet is None:
            control_image = None
        else:
            control_image = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(control_image)
            draw.ellipse(
                (width // 4, height // 4, width // 4 * 3, height // 4 * 3),
                fill=(255, 255, 255),
            )
            del draw

    else:
        control_image = load_image(control_image)
        control_image = control_image.resize((width, height), Image.LANCZOS)
    return control_image


def get_kwarg_inputs(
    prompt,
    negative_prompt,
    height,
    width,
    steps,
    batch,
    seed,
    extra_call_kwargs,
    deepcache,
    cache_interval,
    cache_layer_id,
    cache_block_id,
    input_image,
    control_image,
):
    kwarg_inputs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        num_images_per_prompt=batch,
        generator=(
            None if seed is None else torch.Generator(device="cuda").manual_seed(seed)
        ),
        **(dict() if extra_call_kwargs is None else json.loads(extra_call_kwargs)),
    )
    if input_image is not None:
        kwarg_inputs["image"] = input_image
    if control_image is not None:
        if input_image is None:
            kwarg_inputs["image"] = control_image
        else:
            kwarg_inputs["control_image"] = control_image
    if deepcache:
        kwarg_inputs["cache_interval"] = cache_interval
        kwarg_inputs["cache_layer_id"] = cache_layer_id
        kwarg_inputs["cache_block_id"] = cache_block_id
    return kwarg_inputs


def get_kwarg_inputs_svd(
    input_image,
    height,
    width,
    steps,
    batch,
    deepcache,
    frames,
    fps,
    motion_bucket_id,
    decode_chunk_size,
    seed,
    extra_call_kwargs,
    control_image,
    cache_interval,
    cache_branch,
):
    kwarg_inputs = dict(
        image=input_image,
        height=height,
        width=width,
        num_inference_steps=steps,
        num_videos_per_prompt=batch,
        num_frames=frames,
        fps=fps,
        motion_bucket_id=motion_bucket_id,
        decode_chunk_size=decode_chunk_size,
        generator=(None if seed is None else torch.Generator().manual_seed(seed)),
        **(dict() if extra_call_kwargs is None else json.loads(extra_call_kwargs)),
    )
    if control_image is not None:
        kwarg_inputs["control_image"] = control_image
    if deepcache:
        kwarg_inputs["cache_interval"] = cache_interval
        kwarg_inputs["cache_branch"] = cache_branch
    # remove None values from the dictionary
    kwarg_inputs = {k: v for k, v in kwarg_inputs.items() if v is not None}
    return kwarg_inputs


class IterationProfiler:
    def __init__(self):
        self.begin = None
        self.end = None
        self.num_iterations = 0

    def get_iter_per_sec(self):
        if self.begin is None or self.end is None:
            return None
        self.end.synchronize()
        dur = self.begin.elapsed_time(self.end)
        return self.num_iterations / dur * 1000.0

    def callback_on_step_end(self, pipe, i, t, callback_kwargs={}):
        if self.begin is None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.begin = event
        else:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.end = event
            self.num_iterations += 1
        return callback_kwargs


def calculate_inference_time_and_throughput(prompt, height, width, n_steps, model):
    start_time = time.time()
    model(prompt=prompt, height=height, width=width, num_inference_steps=n_steps)
    end_time = time.time()
    inference_time = end_time - start_time
    # pixels_processed = height * width * n_steps
    # throughput = pixels_processed / inference_time
    throughput = n_steps / inference_time
    return inference_time, throughput


def generate_data_and_fit_model(prompt, height, width, steps_range, model):
    data = {"steps": [], "inference_time": [], "throughput": []}
    print(f"Fitting model...height:{height},width:{width}")
    for n_steps in steps_range:
        inference_time, throughput = calculate_inference_time_and_throughput(
            prompt, height, width, n_steps, model
        )
        data["steps"].append(n_steps)
        data["inference_time"].append(inference_time)
        data["throughput"].append(throughput)
        print(
            f"Steps: {n_steps}, Inference Time: {inference_time:.2f} seconds, Throughput: {throughput:.2f} steps/s"
        )
    average_throughput = np.mean(data["throughput"])
    print(f"Average Throughput: {average_throughput:.2f} steps/s")

    coefficients = np.polyfit(data["steps"], data["inference_time"], 1)
    base_time_without_base_cost = 1 / coefficients[0]
    print(f"Throughput without base cost: {base_time_without_base_cost:.2f} steps/s")
    print(
        f"Model: Inference Time = {coefficients[0]:.2f} * Steps + {coefficients[1]:.2f}"
    )
    return average_throughput, base_time_without_base_cost


def plot_data_and_model(data, coefficients, output_path):
    plt.figure(figsize=(10, 5))
    plt.scatter(data["steps"], data["inference_time"], color="blue")
    plt.plot(data["steps"], np.polyval(coefficients, data["steps"]), color="red")
    plt.title("Inference Time vs. Steps")
    plt.xlabel("Steps")
    plt.ylabel("Inference Time (seconds)")
    plt.grid(True)
    plt.savefig("output.png")
    plt.show()
