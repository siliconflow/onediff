# Run with ONEFLOW_RUN_GRAPH_BY_VM=1 to get faster
MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"
VARIANT = None
CUSTOM_PIPELINE = None
SCHEDULER = None
LORA = None
CONTROLNET = None
STEPS = 25
SEED = None
WARMUPS = 1
FRAMES = None
BATCH = 1
HEIGHT = None
WIDTH = None
FPS = 7
DECODE_CHUNK_SIZE = 5
INPUT_IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true"
EXTRA_CALL_KWARGS = None
ATTENTION_FP16_SCORE_ACCUM_MAX_M = 0
CACHE_INTERVAL = 3
CACHE_BRANCH = 0

import os
import importlib
import inspect
import argparse
import time
import json
from PIL import Image, ImageDraw
import torch
from diffusers.utils import load_image, export_to_video
import oneflow as flow
from onediff.infer_compiler import oneflow_compile
from onediff.infer_compiler.utils import set_boolean_env_var


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--custom-pipeline", type=str, default=CUSTOM_PIPELINE)
    parser.add_argument("--scheduler", type=str, default=SCHEDULER)
    parser.add_argument("--lora", type=str, default=LORA)
    parser.add_argument("--controlnet", type=str, default=None)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--decode-chunk-size",
                        type=int,
                        default=DECODE_CHUNK_SIZE)
    parser.add_argument('--cache_interval', type=int, default=CACHE_INTERVAL)
    parser.add_argument('--cache_branch', type=int, default=CACHE_BRANCH)
    parser.add_argument("--extra-call-kwargs",
                        type=str,
                        default=EXTRA_CALL_KWARGS)
    parser.add_argument(
        "--deepcache",
        type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
        default=False,
    )
    parser.add_argument("--input-image", type=str, default=INPUT_IMAGE)
    parser.add_argument("--control-image", type=str, default=None)
    parser.add_argument("--output-video", type=str, default=None)
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

        controlnet = ControlNetModel.from_pretrained(controlnet,
                                                     torch_dtype=torch.float16)
        extra_kwargs["controlnet"] = controlnet
    is_quantized_model = False
    if os.path.exists(os.path.join(model_name, 'calibrate_info.txt')):
        is_quantized_model = True
        from onediff.quantization import setup_onediff_quant
        setup_onediff_quant()
    pipe = pipeline_cls.from_pretrained(model_name,
                                        torch_dtype=torch.float16,
                                        **extra_kwargs)
    if scheduler is not None:
        scheduler_cls = getattr(importlib.import_module("diffusers"),
                                scheduler)
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    if lora is not None:
        pipe.load_lora_weights(lora)
        pipe.fuse_lora()
    pipe.safety_checker = None
    pipe.to(torch.device("cuda"))

    # Replace quantizable modules by QuantModule.
    if is_quantized_model:
        from onediff.quantization import load_calibration_and_quantize_pipeline
        load_calibration_and_quantize_pipeline(os.path.join(model_name, 'calibrate_info.txt'), pipe)
    return pipe


def compile_pipe(pipe, attention_fp16_score_accum_max_m=-1):
    # set_boolean_env_var('ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_ACCUMULATION',
    #                     False)
    # The absolute element values of K in the attention layer of SVD is too large.
    # The unfused attention (without SDPA) and MHA with half accumulation would both overflow.
    # But disabling all half accumulations in MHA would slow down the inference,
    # especially for 40xx series cards.
    # So here by partially disabling the half accumulation in MHA partially,
    # we can get a good balance.
    set_boolean_env_var(
        "ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_SCORE_ACCUMULATION_MAX_M",
        attention_fp16_score_accum_max_m,
    )

    parts = [
        'image_encoder',
        'unet',
        'controlnet',
    ]
    for part in parts:
        if getattr(pipe, part, None) is not None:
            print(f'Compiling {part}')
            setattr(pipe, part, oneflow_compile(getattr(pipe, part)))
    vae_parts = [
        'decoder',
        'encoder',
    ]
    for part in vae_parts:
        if getattr(pipe.vae, part, None) is not None:
            print(f'Compiling vae.{part}')
            setattr(pipe.vae, part, oneflow_compile(getattr(pipe.vae, part)))
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

    def callback_on_step_end(self, pipe, i, t, callback_kwargs):
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
        from diffusers_extensions.deep_cache import StableVideoDiffusionPipeline
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
        pipe = compile_pipe(
            pipe,
            attention_fp16_score_accum_max_m=args.
            attention_fp16_score_accum_max_m,
        )
    elif args.compiler == "compile":
        pipe.unet = torch.compile(pipe.unet)
        if hasattr(pipe, "controlnet"):
            pipe.controlnet = torch.compile(pipe.controlnet)
        # model.vae = torch.compile(model.vae)
    else:
        raise ValueError(f"Unknown compiler: {args.compiler}")

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
        control_image = Image.open(args.control_image).convert("RGB")
        control_image = control_image.resize((args.width, height),
                                             Image.LANCZOS)

    def get_kwarg_inputs():
        kwarg_inputs = dict(
            image=input_image,
            height=height,
            width=args.width,
            num_inference_steps=args.steps,
            num_videos_per_prompt=args.batch,
            num_frames=args.frames,
            fps=args.fps,
            decode_chunk_size=args.decode_chunk_size,
            generator=None
            if args.seed is None else torch.Generator().manual_seed(args.seed),
            **(dict() if args.extra_call_kwargs is None else json.loads(
                args.extra_call_kwargs)),
        )
        if args.deepcache:
            kwarg_inputs["cache_interval"] = args.cache_interval
            kwarg_inputs["cache_branch"] = args.cache_branch
        if control_image is not None:
            kwarg_inputs["control_image"] = control_image
        return kwarg_inputs

    if args.warmups > 0:
        print("Begin warmup")
        for _ in range(args.warmups):
            pipe(**get_kwarg_inputs())
        print("End warmup")

    kwarg_inputs = get_kwarg_inputs()
    iter_profiler = None
    if "callback_on_step_end" in inspect.signature(pipe).parameters:
        iter_profiler = IterationProfiler()
        kwarg_inputs[
            "callback_on_step_end"] = iter_profiler.callback_on_step_end
    begin = time.time()
    output_frames = pipe(**kwarg_inputs).frames
    end = time.time()

    print(f"Inference time: {end - begin:.3f}s")
    iter_per_sec = iter_profiler.get_iter_per_sec()
    if iter_per_sec is not None:
        print(f"Iterations per second: {iter_per_sec:.3f}")
    cuda_mem_after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
    host_mem_after_used = flow._oneflow_internal.GetCPUMemoryUsed()
    print(f'CUDA Mem after: {cuda_mem_after_used / 1024:.3f}GiB')
    print(f'Host Mem after: {host_mem_after_used / 1024:.3f}GiB')

    if args.output_video is not None:
        export_to_video(output_frames[0], args.output_video, fps=args.fps)
    else:
        print("Please set `--output-video` to save the output video")


if __name__ == "__main__":
    main()
