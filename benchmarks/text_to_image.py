MODEL = "runwayml/stable-diffusion-v1-5"
VARIANT = None
CUSTOM_PIPELINE = None
SCHEDULER = "EulerAncestralDiscreteScheduler"
LORA = None
CONTROLNET = None
STEPS = 30
PROMPT = "best quality, realistic, unreal engine, 4K, a beautiful girl"
NEGATIVE_PROMPT = ""
SEED = 333
WARMUPS = 1
BATCH = 1
HEIGHT = None
WIDTH = None
INPUT_IMAGE = None
CONTROL_IMAGE = None
OUTPUT_IMAGE = None
EXTRA_CALL_KWARGS = None
CACHE_INTERVAL = 3
CACHE_LAYER_ID = 0
CACHE_BLOCK_ID = 0
COMPILER = "oneflow"
COMPILER_CONFIG = None
QUANTIZE_CONFIG = None

import argparse
import importlib
import inspect
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers.utils import load_image
from onediff.infer_compiler import oneflow_compile

from onediffx import (  # quantize_pipe currently only supports the nexfort backend.
    compile_pipe,
    quantize_pipe,
)
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--custom-pipeline", type=str, default=CUSTOM_PIPELINE)
    parser.add_argument("--scheduler", type=str, default=SCHEDULER)
    parser.add_argument("--lora", type=str, default=LORA)
    parser.add_argument("--controlnet", type=str, default=CONTROLNET)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--negative-prompt", type=str, default=NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--cache_interval", type=int, default=CACHE_INTERVAL)
    parser.add_argument("--cache_layer_id", type=int, default=CACHE_LAYER_ID)
    parser.add_argument("--cache_block_id", type=int, default=CACHE_BLOCK_ID)
    parser.add_argument("--extra-call-kwargs", type=str, default=EXTRA_CALL_KWARGS)
    parser.add_argument("--input-image", type=str, default=INPUT_IMAGE)
    parser.add_argument("--control-image", type=str, default=CONTROL_IMAGE)
    parser.add_argument("--output-image", type=str, default=OUTPUT_IMAGE)
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--deepcache", action="store_true")
    parser.add_argument(
        "--compiler",
        type=str,
        default=COMPILER,
        choices=["none", "oneflow", "nexfort", "compile", "compile-max-autotune"],
    )
    parser.add_argument(
        "--compiler-config",
        type=str,
        default=COMPILER_CONFIG,
    )
    parser.add_argument(
        "--run_multiple_resolutions",
        type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
        default=False,
    )
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument(
        "--quantize-config",
        type=str,
        default=QUANTIZE_CONFIG,
    )
    parser.add_argument("--quant-submodules-config-path", type=str, default=None)
    return parser.parse_args()


args = parse_args()


def load_pipe(
    pipeline_cls,
    model_name,
    variant=None,
    dtype=torch.float16,
    device="cuda",
    custom_pipeline=None,
    scheduler=None,
    lora=None,
    controlnet=None,
):
    extra_kwargs = {}
    if custom_pipeline is not None:
        extra_kwargs["custom_pipeline"] = custom_pipeline
    if variant is not None:
        extra_kwargs["variant"] = variant
    if dtype is not None:
        extra_kwargs["torch_dtype"] = dtype
    if controlnet is not None:
        from diffusers import ControlNetModel

        controlnet = ControlNetModel.from_pretrained(
            controlnet,
            torch_dtype=dtype,
        )
        extra_kwargs["controlnet"] = controlnet
    if os.path.exists(os.path.join(model_name, "calibrate_info.txt")):
        from onediff.quantization import QuantPipeline

        pipe = QuantPipeline.from_quantized(pipeline_cls, model_name, **extra_kwargs)
    else:
        pipe = pipeline_cls.from_pretrained(model_name, **extra_kwargs)
    if scheduler is not None and scheduler != "none":
        scheduler_cls = getattr(importlib.import_module("diffusers"), scheduler)
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    if lora is not None:
        pipe.load_lora_weights(lora)
        pipe.fuse_lora()
    pipe.safety_checker = None
    if device is not None:
        pipe.to(torch.device(device))
    return pipe


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


def calculate_inference_time_and_throughput(height, width, n_steps, model):
    start_time = time.time()
    model(prompt=args.prompt, height=height, width=width, num_inference_steps=n_steps)
    end_time = time.time()
    inference_time = end_time - start_time
    # pixels_processed = height * width * n_steps
    # throughput = pixels_processed / inference_time
    throughput = n_steps / inference_time
    return inference_time, throughput


def generate_data_and_fit_model(model, steps_range):
    height, width = 1024, 1024
    data = {"steps": [], "inference_time": [], "throughput": []}

    for n_steps in steps_range:
        inference_time, throughput = calculate_inference_time_and_throughput(
            height, width, n_steps, model
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
    return data, coefficients


def plot_data_and_model(data, coefficients):
    plt.figure(figsize=(10, 5))
    plt.scatter(data["steps"], data["inference_time"], color="blue")
    plt.plot(data["steps"], np.polyval(coefficients, data["steps"]), color="red")
    plt.title("Inference Time vs. Steps")
    plt.xlabel("Steps")
    plt.ylabel("Inference Time (seconds)")
    plt.grid(True)
    # plt.savefig("output.png")
    plt.show()

    print(
        f"Model: Inference Time = {coefficients[0]:.2f} * Steps + {coefficients[1]:.2f}"
    )


def main():
    if args.input_image is None:
        if args.deepcache:
            from onediffx.deep_cache import StableDiffusionXLPipeline as pipeline_cls
        else:
            from diffusers import AutoPipelineForText2Image as pipeline_cls
    else:
        from diffusers import AutoPipelineForImage2Image as pipeline_cls

    pipe = load_pipe(
        pipeline_cls,
        args.model,
        variant=args.variant,
        custom_pipeline=args.custom_pipeline,
        scheduler=args.scheduler,
        lora=args.lora,
        controlnet=args.controlnet,
    )

    core_net = None
    if core_net is None:
        core_net = getattr(pipe, "unet", None)
    if core_net is None:
        core_net = getattr(pipe, "transformer", None)
    height = args.height or core_net.config.sample_size * pipe.vae_scale_factor
    width = args.width or core_net.config.sample_size * pipe.vae_scale_factor

    if args.compiler == "none":
        pass
    elif args.compiler == "oneflow":
        print("Oneflow backend is now active...")
        # Note: The compile_pipe() based on the oneflow backend is incompatible with T5EncoderModel.
        # pipe = compile_pipe(pipe)
        if hasattr(pipe, "unet"):
            pipe.unet = oneflow_compile(pipe.unet)
        if hasattr(pipe, "transformer"):
            pipe.transformer = oneflow_compile(pipe.transformer)
        pipe.vae.decoder = oneflow_compile(pipe.vae.decoder)
    elif args.compiler == "nexfort":
        print("Nexfort backend is now active...")
        if args.quantize:
            if args.quantize_config is not None:
                quantize_config = json.loads(args.quantize_config)
            else:
                quantize_config = '{"quant_type": "fp8_e4m3_e4m3_dynamic"}'
            if args.quant_submodules_config_path:
                # download: https://huggingface.co/siliconflow/PixArt-alpha-onediff-nexfort-fp8/blob/main/fp8_e4m3.json
                pipe = quantize_pipe(
                    pipe,
                    quant_submodules_config_path=args.quant_submodules_config_path,
                    ignores=[],
                    **quantize_config,
                )
            else:
                pipe = quantize_pipe(pipe, ignores=[], **quantize_config)
        if args.compiler_config is not None:
            # config with dict
            options = json.loads(args.compiler_config)
        else:
            # config with string
            options = '{"mode": "max-optimize:max-autotune:low-precision", "memory_format": "channels_last"}'
        pipe = compile_pipe(
            pipe, backend="nexfort", options=options, fuse_qkv_projections=True
        )
    elif args.compiler in ("compile", "compile-max-autotune"):
        mode = "max-autotune" if args.compiler == "compile-max-autotune" else None
        if hasattr(pipe, "unet"):
            pipe.unet = torch.compile(pipe.unet, mode=mode)
        if hasattr(pipe, "transformer"):
            pipe.transformer = torch.compile(pipe.transformer, mode=mode)
        if hasattr(pipe, "controlnet"):
            pipe.controlnet = torch.compile(pipe.controlnet, mode=mode)
        pipe.vae = torch.compile(pipe.vae, mode=mode)
    else:
        raise ValueError(f"Unknown compiler: {args.compiler}")

    if args.input_image is None:
        input_image = None
    else:
        input_image = load_image(args.input_image)
        input_image = input_image.resize((width, height), Image.LANCZOS)

    if args.control_image is None:
        if args.controlnet is None:
            control_image = None
        else:
            control_image = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(control_image)
            draw.ellipse(
                (args.width // 4, height // 4, args.width // 4 * 3, height // 4 * 3),
                fill=(255, 255, 255),
            )
            del draw
    else:
        control_image = load_image(args.control_image)
        control_image = control_image.resize((width, height), Image.LANCZOS)

    def get_kwarg_inputs():
        kwarg_inputs = dict(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=height,
            width=width,
            num_images_per_prompt=args.batch,
            generator=(
                None
                if args.seed is None
                else torch.Generator(device="cuda").manual_seed(args.seed)
            ),
            **(
                dict()
                if args.extra_call_kwargs is None
                else json.loads(args.extra_call_kwargs)
            ),
        )
        if args.steps is not None:
            kwarg_inputs["num_inference_steps"] = args.steps
        if input_image is not None:
            kwarg_inputs["image"] = input_image
        if control_image is not None:
            if input_image is None:
                kwarg_inputs["image"] = control_image
            else:
                kwarg_inputs["control_image"] = control_image
        if args.deepcache:
            kwarg_inputs["cache_interval"] = args.cache_interval
            kwarg_inputs["cache_layer_id"] = args.cache_layer_id
            kwarg_inputs["cache_block_id"] = args.cache_block_id
        return kwarg_inputs

    # NOTE: Warm it up.
    # The initial calls will trigger compilation and might be very slow.
    # After that, it should be very fast.
    if args.warmups > 0:
        begin = time.time()
        print("=======================================")
        print("Begin warmup")
        for _ in range(args.warmups):
            pipe(**get_kwarg_inputs())
        end = time.time()
        print("End warmup")
        print(f"Warmup time: {end - begin:.3f}s")
        print("=======================================")

    # Let"s see it!
    # Note: Progress bar might work incorrectly due to the async nature of CUDA.
    kwarg_inputs = get_kwarg_inputs()
    iter_profiler = IterationProfiler()
    if "callback_on_step_end" in inspect.signature(pipe).parameters:
        kwarg_inputs["callback_on_step_end"] = iter_profiler.callback_on_step_end
    elif "callback" in inspect.signature(pipe).parameters:
        kwarg_inputs["callback"] = iter_profiler.callback_on_step_end
    begin = time.time()
    output_images = pipe(**kwarg_inputs).images
    end = time.time()

    print("=======================================")
    print(f"Inference time: {end - begin:.3f}s")
    iter_per_sec = iter_profiler.get_iter_per_sec()
    if iter_per_sec is not None:
        print(f"Iterations per second: {iter_per_sec:.3f}")
    if args.compiler == "oneflow":
        import oneflow as flow  # usort: skip

        cuda_mem_after_used = flow._oneflow_internal.GetCUDAMemoryUsed() / 1024
    else:
        cuda_mem_after_used = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Max used CUDA memory : {cuda_mem_after_used:.3f}GiB")
    print("=======================================")

    if args.print_output:
        from onediff.utils.import_utils import is_nexfort_available

        if is_nexfort_available():
            from nexfort.utils.term_image import print_image

            for image in output_images:
                print_image(image, max_width=80)

    if args.output_image is not None:
        output_images[0].save(args.output_image)
    else:
        print("Please set `--output-image` to save the output image")

    if args.run_multiple_resolutions:
        print("Test run with multiple resolutions...")
        sizes = [1024, 512, 768, 256]
        for h in sizes:
            for w in sizes:
                kwarg_inputs["height"] = h
                kwarg_inputs["width"] = w
                print(f"Running at resolution: {h}x{w}")
                start_time = time.time()
                image = pipe(**kwarg_inputs).images
                end_time = time.time()
                print(f"Inference time: {end_time - start_time:.2f} seconds")

    if args.throughput:
        steps_range = range(1, 100, 1)
        data, coefficients = generate_data_and_fit_model(pipe, steps_range)
        plot_data_and_model(data, coefficients)


if __name__ == "__main__":
    main()
