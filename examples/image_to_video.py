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
HEIGHT = 576
WIDTH = 1024
FPS = 7
DECODE_CHUNK_SIZE = 4
INPUT_IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true"
EXTRA_CALL_KWARGS = None
ATTENTION_FP16_SCORE_ACCUM_MAX_M = 0

import importlib
import inspect
import argparse
import time
import json
from PIL import Image, ImageDraw

import oneflow as flow
import torch

from diffusers.utils import load_image, export_to_video
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
    parser.add_argument("--decode-chunk-size", type=int, default=DECODE_CHUNK_SIZE)
    parser.add_argument("--extra-call-kwargs", type=str, default=EXTRA_CALL_KWARGS)
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


def load_model(
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
            controlnet, torch_dtype=torch.float16
        )
        extra_kwargs["controlnet"] = controlnet
    model = pipeline_cls.from_pretrained(
        model_name, torch_dtype=torch.float16, **extra_kwargs
    )
    if scheduler is not None:
        scheduler_cls = getattr(importlib.import_module("diffusers"), scheduler)
        model.scheduler = scheduler_cls.from_config(model.scheduler.config)
    if lora is not None:
        model.load_lora_weights(lora)
        model.fuse_lora()
    model.safety_checker = None
    model.to(torch.device("cuda"))
    return model


def compile_model(model, attention_fp16_score_accum_max_m=-1):
    # set_boolean_env_var('ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_ACCUMULATION',
    #                     False)
    # The absolute element values of K in the attention layer of SVD is too large.
    # The unfused attention (without SDPA) and MHA with half accumulation would both overflow.
    # But disabling all half accumulations in MHA would slow down the inference,
    # especially for 40xx series cards.
    # So here by partially disabling the half accumulation in MHA partially,
    # we can get a good balance.
    #
    # On RTX 4090:
    # | ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_ACCUMULATION | MAX_M | Output | Duration |
    # | --------------------------------------------------- | ------| ------ | -------- |
    # | False                                               | -1    | OK     | 32.251s  |
    # | True                                                | -1    | NaN    | 29.089s  |
    # | True                                                | 0     | OK     | 30.947s  |
    # | True                                                | 2304  | OK     | 30.820s  |
    set_boolean_env_var(
        "ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_SCORE_ACCUMULATION_MAX_M",
        attention_fp16_score_accum_max_m,
    )

    model.image_encoder = oneflow_compile(model.image_encoder)
    model.unet = oneflow_compile(model.unet)
    model.vae.decoder = oneflow_compile(model.vae.decoder)
    model.vae.encoder = oneflow_compile(model.vae.encoder)
    return model


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
    from diffusers import StableVideoDiffusionPipeline

    model = load_model(
        StableVideoDiffusionPipeline,
        args.model,
        variant=args.variant,
        custom_pipeline=args.custom_pipeline,
        scheduler=args.scheduler,
        lora=args.lora,
        controlnet=args.controlnet,
    )

    if args.compiler == "none":
        pass
    elif args.compiler == "oneflow":
        model = compile_model(
            model,
            attention_fp16_score_accum_max_m=args.attention_fp16_score_accum_max_m,
        )
    elif args.compiler == "compile":
        model.unet = torch.compile(model.unet)
        if hasattr(model, "controlnet"):
            model.controlnet = torch.compile(model.controlnet)
        # model.vae = torch.compile(model.vae)
    else:
        raise ValueError(f"Unknown compiler: {args.compiler}")

    input_image = load_image(args.input_image)
    input_image.resize((args.width, args.height), Image.LANCZOS)

    if args.control_image is None:
        if args.controlnet is None:
            control_image = None
        else:
            control_image = Image.new("RGB", (args.width, args.height))
            draw = ImageDraw.Draw(control_image)
            draw.ellipse(
                (
                    args.width // 4,
                    args.height // 4,
                    args.width // 4 * 3,
                    args.height // 4 * 3,
                ),
                fill=(255, 255, 255),
            )
            del draw
    else:
        control_image = Image.open(args.control_image).convert("RGB")
        control_image = control_image.resize((args.width, args.height), Image.LANCZOS)

    def get_kwarg_inputs():
        kwarg_inputs = dict(
            image=input_image,
            height=args.height,
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
        if control_image is not None:
            kwarg_inputs["control_image"] = control_image
        return kwarg_inputs

    if args.warmups > 0:
        print("Begin warmup")
        for _ in range(args.warmups):
            model(**get_kwarg_inputs())
        print("End warmup")

    kwarg_inputs = get_kwarg_inputs()
    iter_profiler = None
    if "callback_on_step_end" in inspect.signature(model).parameters:
        iter_profiler = IterationProfiler()
        kwarg_inputs["callback_on_step_end"] = iter_profiler.callback_on_step_end
    if args.seed is not None:
        torch.manual_seed(args.seed)
    begin = time.time()
    output_frames = model(**kwarg_inputs).frames
    end = time.time()

    print(f"Inference time: {end - begin:.3f}s")
    iter_per_sec = iter_profiler.get_iter_per_sec()
    if iter_per_sec is not None:
        print(f"Iterations per second: {iter_per_sec:.3f}")
    peak_mem = torch.cuda.max_memory_allocated()
    print(f"Peak memory: {peak_mem / 1024**3:.3f}GiB")

    if args.output_video is not None:
        export_to_video(output_frames[0], args.output_video, fps=args.fps)
    else:
        print("Please set `--output-video` to save the output-video")


if __name__ == "__main__":
    main()
