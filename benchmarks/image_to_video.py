MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"
# SVD 1.1: stabilityai/stable-video-diffusion-img2vid-xt-1-1 is also available.
VARIANT = None
CUSTOM_PIPELINE = None
SCHEDULER = None
LORA = None
CONTROLNET = None
STEPS = 25
SEED = None
WARMUPS = 1
BATCH = 1
ALTER_HEIGHT = None
ALTER_WIDTH = None
# The official recommended parameters for SVD 1.1 are:
# Resolution: 1024x576,
# Frame count: 25 frames,
# FPS: 6,
# Motion Bucket Id: 127.
HEIGHT = 576
WIDTH = 1024
FRAMES = 25
FPS = 7
MOTION_BUCKET_ID = 127
DECODE_CHUNK_SIZE = 5
INPUT_IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true"
CONTROL_IMAGE = None
OUTPUT_VIDEO = None
EXTRA_CALL_KWARGS = None
ATTENTION_FP16_SCORE_ACCUM_MAX_M = 0
CACHE_INTERVAL = 3
CACHE_BRANCH = 0

import argparse
import importlib
import inspect
import json
import os
import random
import time

from PIL import Image, ImageDraw

import oneflow as flow  # usort: skip
import torch
from diffusers.utils import export_to_video, load_image
from onediffx import compile_pipe, OneflowCompileOptions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--save_graph", action="store_true")
    parser.add_argument("--load_graph", action="store_true")
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--custom-pipeline", type=str, default=CUSTOM_PIPELINE)
    parser.add_argument("--scheduler", type=str, default=SCHEDULER)
    parser.add_argument("--lora", type=str, default=LORA)
    parser.add_argument("--controlnet", type=str, default=CONTROLNET)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--motion_bucket_id", type=int, default=MOTION_BUCKET_ID)
    parser.add_argument("--decode-chunk-size", type=int, default=DECODE_CHUNK_SIZE)
    parser.add_argument("--cache_interval", type=int, default=CACHE_INTERVAL)
    parser.add_argument("--cache_branch", type=int, default=CACHE_BRANCH)
    parser.add_argument("--extra-call-kwargs", type=str, default=EXTRA_CALL_KWARGS)
    parser.add_argument("--deepcache", action="store_true")
    parser.add_argument("--input-image", type=str, default=INPUT_IMAGE)
    parser.add_argument("--control-image", type=str, default=CONTROL_IMAGE)
    parser.add_argument("--output-video", type=str, default=OUTPUT_VIDEO)
    parser.add_argument(
        "--compiler",
        type=str,
        default="oneflow",
        choices=["none", "oneflow", "compile"],
    )
    parser.add_argument(
        "--attention-fp16-score-accum-max-m",
        type=int,
        default=ATTENTION_FP16_SCORE_ACCUM_MAX_M,
    )
    parser.add_argument(
        "--alter-height",
        type=int,
        default=ALTER_HEIGHT,
    )
    parser.add_argument(
        "--alter-width",
        type=int,
        default=ALTER_WIDTH,
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

        pipe = QuantPipeline.from_quantized(
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
    if args.deepcache:
        from onediffx.deep_cache import StableVideoDiffusionPipeline
    else:
        from diffusers import StableVideoDiffusionPipeline

    pipe = load_pipe(
        StableVideoDiffusionPipeline,
        args.model,
        variant=args.variant,
        custom_pipeline=args.custom_pipeline,
        scheduler=args.scheduler,
        lora=args.lora,
        controlnet=args.controlnet,
    )

    height = args.height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = args.width or pipe.unet.config.sample_size * pipe.vae_scale_factor

    if args.compiler == "none":
        pass
    elif args.compiler == "oneflow":
        # The absolute element values of K in the attention layer of SVD is too large.
        # The unfused attention (without SDPA) and MHA with half accumulation would both overflow.
        # But disabling all half accumulations in MHA would slow down the inference,
        # especially for 40xx series cards.
        # So here by partially disabling the half accumulation in MHA partially,
        # we can get a good balance.
        compile_options = OneflowCompileOptions()
        compile_options.attention_allow_half_precision_score_accumulation_max_m = (
            args.attention_fp16_score_accum_max_m
        )
        pipe = compile_pipe(pipe, options=compile_options)
    elif args.compiler == "compile":
        pipe.unet = torch.compile(pipe.unet)
        if hasattr(pipe, "controlnet"):
            pipe.controlnet = torch.compile(pipe.controlnet)
        pipe.vae = torch.compile(pipe.vae)
    else:
        raise ValueError(f"Unknown compiler: {args.compiler}")

    resolutions = [[height, width]]
    if args.alter_height is not None:
        # Test dynamic shape.
        assert args.alter_width is not None
        resolutions.append([args.alter_height, args.alter_width])
    for height, width in resolutions:
        input_image = load_image(args.input_image)
        input_image.resize((width, height), Image.LANCZOS)

        if args.control_image is None:
            if args.controlnet is None:
                control_image = None
            else:
                control_image = Image.new("RGB", (width, height))
                draw = ImageDraw.Draw(control_image)
                draw.ellipse(
                    (
                        width // 4,
                        height // 4,
                        width // 4 * 3,
                        height // 4 * 3,
                    ),
                    fill=(255, 255, 255),
                )
                del draw
        else:
            control_image = load_image(args.control_image)
            control_image = control_image.resize((args.width, height), Image.LANCZOS)

        def get_kwarg_inputs():
            kwarg_inputs = dict(
                image=input_image,
                height=height,
                width=width,
                num_inference_steps=args.steps,
                num_videos_per_prompt=args.batch,
                num_frames=args.frames,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                decode_chunk_size=args.decode_chunk_size,
                generator=None
                if args.seed is None
                else torch.Generator().manual_seed(args.seed),
                **(
                    dict()
                    if args.extra_call_kwargs is None
                    else json.loads(args.extra_call_kwargs)
                ),
            )
            if control_image is not None:
                kwarg_inputs["control_image"] = control_image
            if args.deepcache:
                kwarg_inputs["cache_interval"] = args.cache_interval
                kwarg_inputs["cache_branch"] = args.cache_branch
            return kwarg_inputs

        if args.warmups > 0:
            if args.load_graph:
                print("Loading graphs to avoid compilation...")
                start_t = time.time()
                pipe.unet.load_graph("base_unet_compiled", run_warmup=True)
                pipe.vae.decoder.load_graph("base_vae_compiled", run_warmup=True)
                end_t = time.time()
                print(f"Loading graph elapsed: {end_t - start_t} s")
                print("Begin warmup")
                for _ in range(args.warmups):
                    pipe(**get_kwarg_inputs())
                print("End warmup")
            else:
                print("Begin warmup")
                for _ in range(args.warmups):
                    pipe(**get_kwarg_inputs())
                print("End warmup")

        kwarg_inputs = get_kwarg_inputs()
        iter_profiler = IterationProfiler()
        if "callback_on_step_end" in inspect.signature(pipe).parameters:
            kwarg_inputs["callback_on_step_end"] = iter_profiler.callback_on_step_end
        elif "callback" in inspect.signature(pipe).parameters:
            kwarg_inputs["callback"] = iter_profiler.callback_on_step_end
        begin = time.time()
        output_frames = pipe(**kwarg_inputs).frames
        end = time.time()

        print(f"Inference time: {end - begin:.3f}s")
        iter_per_sec = iter_profiler.get_iter_per_sec()
        if iter_per_sec is not None:
            print(f"Iterations per second: {iter_per_sec:.3f}")
        cuda_mem_after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
        host_mem_after_used = flow._oneflow_internal.GetCPUMemoryUsed()
        print(f"CUDA Mem after: {cuda_mem_after_used / 1024:.3f}GiB")
        print(f"Host Mem after: {host_mem_after_used / 1024:.3f}GiB")

        if args.output_video is not None:
            export_to_video(output_frames[0], args.output_video, fps=args.fps)
        else:
            print("Please set `--output-video` to save the output video")

    if args.save_graph:
        print("Saving graphs...")
        start_t = time.time()
        pipe.unet.save_graph("base_unet_compiled")
        pipe.vae.decoder.save_graph("base_vae_compiled")
        end_t = time.time()
        print(f"save graphs elapsed: {end_t - start_t} s")


if __name__ == "__main__":
    main()
