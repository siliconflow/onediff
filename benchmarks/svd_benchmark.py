import os
import importlib
import inspect
import argparse
import time
import json
import random
from PIL import Image, ImageDraw

import oneflow as flow
import torch
from onediffx import compile_pipe, compiler_config
from diffusers.utils import load_image, export_to_video

from benchmark_base import BaseBenchmark
from utils.sd_utils import *

INPUT_IMAGE = "/data/home/wangerlie/onediff/benchmarks/resources/rocket.png"


class SVDBenchmark(BaseBenchmark):
    def __init__(
        self,
        model_dir=None,
        model_name="stabilityai/stable-video-diffusion-img2vid-xt",
        save_graph=False,
        load_graph=False,
        variant="fp16",
        device="cuda",
        custom_pipeline=None,
        scheduler=None,
        lora=None,
        controlnet=None,
        steps=25,
        seed=None,
        warmups=1,
        frames=None,
        batch=1,
        height=576,
        width=1024,
        fps=7,
        motion_bucket_id=127,
        decode_chunk_size=5,
        cache_interval=None,
        cache_branch=None,
        extra_call_kwargs=None,
        deepcache=False,
        input_image=INPUT_IMAGE,
        control_image=None,
        output_video=None,
        compiler="oneflow",
        attention_fp16_score_accum_max_m=0,
        alter_height=None,
        alter_width=None,
    ):
        self.model_dir = model_dir
        self.model_name = model_name
        self.save_graph = save_graph
        self.load_graph = load_graph
        self.variant = variant
        self.custom_pipeline = custom_pipeline
        self.scheduler = scheduler
        self.lora = lora
        self.controlnet = controlnet
        self.steps = steps
        self.seed = seed
        self.warmups = warmups
        self.frames = frames
        self.batch = batch
        self.height = height
        self.width = width
        self.fps = fps
        self.motion_bucket_id = motion_bucket_id
        self.decode_chunk_size = decode_chunk_size
        self.cache_interval = cache_interval
        self.cache_branch = cache_branch
        self.extra_call_kwargs = extra_call_kwargs
        self.deepcache = deepcache
        self.input_image = input_image
        self.control_image = control_image
        self.output_video = output_video
        self.compiler = compiler
        self.attention_fp16_score_accum_max_m = attention_fp16_score_accum_max_m
        self.alter_height = alter_height
        self.alter_width = alter_width

        self.device = get_device(device)

    def load_pipeline_from_diffusers(self):
        if self.deepcache:
            from onediffx.deep_cache import StableVideoDiffusionPipeline
        else:
            from diffusers import StableVideoDiffusionPipeline
        if self.model_dir:
            print("Use Local Model.")
            self.model_path = os.path.join(
                self.model_dir, self.model_name.split("/")[-1]
            )
            if os.path.exists(self.model_path):
                self.pipe = load_sd_pipe(
                    pipeline_cls=StableVideoDiffusionPipeline,
                    model_name=self.model_path,
                    dtype=torch.float16,
                    variant=self.variant,
                    custom_pipeline=self.custom_pipeline,
                    scheduler=self.scheduler,
                    lora=self.lora,
                    controlnet=self.controlnet,
                    device=torch.device("cuda"),
                )
            else:
                raise ValueError("Model path {self.model_path} dose not exist")
        else:
            print("Use HF Model")
            self.pipe = load_sd_pipe(
                pipeline_cls=StableVideoDiffusionPipeline,
                model_name=self.model_name,
                variant=self.variant,
                custom_pipeline=self.custom_pipeline,
                scheduler=self.scheduler,
                lora=self.lora,
                controlnet=self.controlnet,
                device=torch.device("cuda"),
            )
        self.height = (
            self.height or pipe.unet.config.sample_size * pipe.vae_scale_factor
        )
        self.width = self.width or pipe.unet.config.sample_size * pipe.vae_scale_factor

    def compile_pipeline(self):
        if self.compiler is None:
            pass
        elif self.compiler == "oneflow":
            self.pipe = compile_pipe(self.pipe)
            print("Compile pipeline with OneFlow")
        elif self.compiler in ("compile", "compile-max-autotune"):
            mode = "max-autotune" if self.compiler == "compile-max-autotune" else None
            self.pipe.unet = torch.compile(self.pipe.unet, mode=mode)
            if hasattr(self.pipe, "controlnet"):
                self.pipe.controlnet = torch.compile(self.pipe.controlnet, mode=mode)
            self.pipe.vae = torch.compile(self.pipe.vae, mode=mode)
        else:
            raise ValueError(f"Unknown compiler: {self.compiler}")

    def benchmark_model(self):
        self.resolutions = [[self.height, self.width]]
        if self.alter_height is not None:
            # Test dynamic shape.
            assert self.alter_width is not None
            self.resolutions.append([self.alter_height, self.alter_width])
        for height, width in self.resolutions:
            self.input_image = load_image(self.input_image)
            self.input_image.resize((width, height), Image.LANCZOS)

            if self.control_image is None:
                if self.controlnet is None:
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
                control_image = load_image(self.control_image)
                control_image = control_image.resize(
                    (self.width, height), Image.LANCZOS
                )
        self.kwarg_inputs = get_kwarg_inputs_svd(
            input_image=self.input_image,
            height=self.height,
            width=self.width,
            steps=self.steps,
            batch=self.batch,
            deepcache=self.deepcache,
            frames=self.frames,
            fps=self.fps,
            motion_bucket_id=self.motion_bucket_id,
            decode_chunk_size=self.decode_chunk_size,
            seed=self.seed,
            extra_call_kwargs=self.extra_call_kwargs,
            control_image=self.control_image,
            cache_interval=self.cache_interval,
            cache_branch=self.cache_branch,
        )
        if self.warmups > 0:
            if self.load_graph:
                print("Loading graphs to avoid compilation...")
                start_t = time.time()
                self.pipe.unet.load_graph("base_unet_compiled", run_warmup=True)
                self.pipe.vae.decoder.load_graph("base_vae_compiled", run_warmup=True)
                end_t = time.time()
                print(f"Loading graph elapsed: {end_t - start_t} s")
                print("Begin warmup")
                for _ in range(self.warmups):
                    self.pipe(**self.kwarg_inputs)
                print("End warmup")
            else:
                print("Begin warmup")
                for _ in range(self.warmups):
                    self.pipe(**self.kwarg_inputs)
                print("End warmup")

        iter_profiler = IterationProfiler()
        if "callback_on_step_end" in inspect.signature(self.pipe).parameters:
            self.kwarg_inputs["callback_on_step_end"] = (
                iter_profiler.callback_on_step_end
            )
        elif "callback" in inspect.signature(self.pipe).parameters:
            self.kwarg_inputs["callback"] = iter_profiler.callback_on_step_end
        begin = time.time()
        output_frames = self.pipe(**self.kwarg_inputs).frames
        end = time.time()
        self.results = {}
        print(f"Inference time: {end - begin:.3f}s")
        self.results["inference_time"] = end - begin
        iter_per_sec = iter_profiler.get_iter_per_sec()
        self.results["iter_per_sec"] = iter_per_sec
        if iter_per_sec is not None:
            print(f"Iterations per second: {iter_per_sec:.3f}")
        cuda_mem_after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
        host_mem_after_used = flow._oneflow_internal.GetCPUMemoryUsed()
        print(f"CUDA Mem after: {cuda_mem_after_used / 1024:.3f}GiB")
        print(f"Host Mem after: {host_mem_after_used / 1024:.3f}GiB")
        self.results["cuda_mem_after_used"] = cuda_mem_after_used / 1024
        self.results["host_mem_after_used"] = host_mem_after_used / 1024

        if self.output_video is not None:
            export_to_video(output_frames[0], self.output_video, fps=self.fps)
        else:
            print("Please set `--output-video` to save the output video")


if __name__ == "__main__":
    benchmark = SVDBenchmark(
        model_dir=r"/data/home/wangerlie/onediff/benchmarks/models"
    )
    benchmark.load_pipeline_from_diffusers()
    benchmark.compile_pipeline()
    benchmark.benchmark_model()
