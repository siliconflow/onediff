# Requirements: diffusers==0.25.1.
# Reference: https://github.com/InstantID/InstantID/blob/main/gradio_demo/requirements.txt
REPO = None
FACE_ANALYSIS_ROOT = None
MODEL = "wangqixun/YamerMIX_v8"
VARIANT = None
CUSTOM_PIPELINE = "pipeline_stable_diffusion_xl_instantid"
SCHEDULER = "EulerAncestralDiscreteScheduler"
LORA = None
CONTROLNET = "InstantX/InstantID"
STEPS = 30
PROMPT = "film noir style, ink sketch|vector, male man, highly detailed, sharp focus, ultra sharpness, monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic"
NEGATIVE_PROMPT = "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vibrant, colorful"
SEED = None
WARMUPS = 3
BATCH = 1
HEIGHT = None
WIDTH = None
INPUT_IMAGE = "https://github.com/InstantID/InstantID/blob/main/examples/musk_resize.jpeg?raw=true"
OUTPUT_IMAGE = None
EXTRA_CALL_KWARGS = """{
    "controlnet_conditioning_scale": 0.8,
    "ip_adapter_scale": 0.8
}"""
CACHE_INTERVAL = 3
CACHE_LAYER_ID = 0
CACHE_BLOCK_ID = 0

import argparse
import importlib
import inspect
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from diffusers.utils import load_image
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw

import oneflow as flow  # usort: skip
from onediffx import compile_pipe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default=REPO)
    parser.add_argument("--face-analysis-root", type=str, default=FACE_ANALYSIS_ROOT)
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
    parser.add_argument("--output-image", type=str, default=OUTPUT_IMAGE)
    parser.add_argument("--deepcache", action="store_true")
    parser.add_argument(
        "--compiler",
        type=str,
        default="oneflow",
        choices=["none", "oneflow", "compile", "compile-max-autotune"],
    )
    return parser.parse_args()


def load_pipe(
    pipeline_cls,
    model_name,
    variant=None,
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
    if controlnet is not None:
        from diffusers import ControlNetModel

        controlnet = ControlNetModel.from_pretrained(
            controlnet,
            torch_dtype=torch.float16,
        )
        extra_kwargs["controlnet"] = controlnet
    if os.path.exists(os.path.join(model_name, "calibrate_info.txt")):
        from onediff.quantization import QuantPipeline

        pipe = QuantPipeline.from_pretrained(
            pipeline_cls, model_name, torch_dtype=torch.float16, **extra_kwargs
        )
    else:
        pipe = pipeline_cls.from_pretrained(
            model_name, torch_dtype=torch.float16, **extra_kwargs
        )
    if scheduler is not None:
        scheduler_cls = getattr(importlib.import_module("diffusers"), scheduler)
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    if lora is not None:
        pipe.load_lora_weights(lora)
        pipe.fuse_lora()
    pipe.safety_checker = None
    pipe.to(torch.device("cuda"))
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


def main():
    args = parse_args()

    assert (
        args.controlnet is not None
    ), "Please set `--controlnet` to the name or path of the controlnet"
    assert (
        args.face_analysis_root is not None
    ), "Please set `--face-analysis-root` to the path of the working directory of insightface.app.FaceAnalysis"
    assert os.path.isdir(
        os.path.join(args.face_analysis_root, "models", "antelopev2")
    ), f"Please download models from https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing and extract to {os.path.join(args.face_analysis_root, 'models', 'antelopev2')}"
    assert (
        args.input_image is not None
    ), "Please set `--input-image` to the path of the input image"
    assert not args.deepcache

    if args.compiler == "oneflow":
        # Patch the attention processors to make them compatible with us.
        attention_processor_path = os.path.join(
            args.repo, "ip_adapter", "attention_processor.py"
        )

        with open(attention_processor_path, "r") as f:
            content = f.read()

        if "__call__" in content:
            content = content.replace("__call__", "forward")
            with open(attention_processor_path, "w") as f:
                f.write(content)

    custom_pipeline = None
    draw_kps = None
    if args.repo is None:
        custom_pipeline = args.custom_pipeline
        from diffusers import DiffusionPipeline

        pipeline_cls = DiffusionPipeline
    else:
        sys.path.insert(0, args.repo)

        from pipeline_stable_diffusion_xl_instantid import (
            draw_kps,
            StableDiffusionXLInstantIDPipeline as pipeline_cls,
        )

    if os.path.exists(args.controlnet):
        controlnet = args.controlnet
    else:
        controlnet = snapshot_download(args.controlnet)

    app = FaceAnalysis(
        name="antelopev2",
        root=args.face_analysis_root,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    face_adapter = os.path.join(controlnet, "ip-adapter.bin")
    controlnet_path = os.path.join(controlnet, "ControlNetModel")

    pipe = load_pipe(
        pipeline_cls,
        args.model,
        variant=args.variant,
        custom_pipeline=custom_pipeline,
        scheduler=args.scheduler,
        lora=args.lora,
        controlnet=controlnet_path,
    )

    if draw_kps is None:
        draw_kps = sys.modules[pipe.__module__].draw_kps

    pipe.load_ip_adapter_instantid(face_adapter)

    height = args.height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = args.width or pipe.unet.config.sample_size * pipe.vae_scale_factor

    if args.compiler == "none":
        pass
    elif args.compiler == "oneflow":
        pipe = compile_pipe(pipe)
    elif args.compiler in ("compile", "compile-max-autotune"):
        mode = "max-autotune" if args.compiler == "compile-max-autotune" else None
        pipe.unet = torch.compile(pipe.unet, mode=mode)
        if hasattr(pipe, "controlnet"):
            pipe.controlnet = torch.compile(pipe.controlnet, mode=mode)
        pipe.vae = torch.compile(pipe.vae, mode=mode)
    else:
        raise ValueError(f"Unknown compiler: {args.compiler}")

    input_image = load_image(args.input_image)
    input_image = input_image.resize((width, height), Image.LANCZOS)

    face_image = input_image
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(
        face_info,
        key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1],
    )[
        -1
    ]  # only use the maximum face
    face_emb = face_info["embedding"]
    face_kps = draw_kps(face_image, face_info["kps"])

    def get_kwarg_inputs():
        kwarg_inputs = dict(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            height=height,
            width=width,
            num_inference_steps=args.steps,
            num_images_per_prompt=args.batch,
            generator=None
            if args.seed is None
            else torch.Generator(device="cuda").manual_seed(args.seed),
            **(
                dict()
                if args.extra_call_kwargs is None
                else json.loads(args.extra_call_kwargs)
            ),
        )
        if args.deepcache:
            kwarg_inputs["cache_interval"] = args.cache_interval
            kwarg_inputs["cache_layer_id"] = args.cache_layer_id
            kwarg_inputs["cache_block_id"] = args.cache_block_id
        return kwarg_inputs

    # NOTE: Warm it up.
    # The initial calls will trigger compilation and might be very slow.
    # After that, it should be very fast.
    if args.warmups > 0:
        print("Begin warmup")
        for _ in range(args.warmups):
            pipe(**get_kwarg_inputs())
        print("End warmup")

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
    cuda_mem_after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    host_mem_after_used = flow._oneflow_internal.GetCPUMemoryUsed()
    print(f"CUDA Mem after: {cuda_mem_after_used / 1024:.3f}GiB")
    print(f"Host Mem after: {host_mem_after_used / 1024:.3f}GiB")
    print("=======================================")

    if args.output_image is not None:
        output_images[0].save(args.output_image)
    else:
        print("Please set `--output-image` to save the output image")


if __name__ == "__main__":
    main()
