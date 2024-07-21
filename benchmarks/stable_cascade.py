# NOTE:
# ValueError: Cannot load /home/fenrir/.cache/huggingface/hub/models--stabilityai--stable-cascade/snapshots/f2a84281d6f8db3c757195dd0c9a38dbdea90bb4/decoder because embedding.1.weight expected shape tensor(..., device='meta', size=(320, 64, 1, 1)), but got torch.Size([320, 16, 1, 1]). If you want to instead overwrite randomly initialized weights, please make sure to pass both low_cpu_mem_usage=False and ignore_mismatched_sizes=True. For more information, see also: https://github.com/huggingface/diffusers/issues/1619#issuecomment-1345604389 as an example.
# Solution is here: https://huggingface.co/stabilityai/stable-cascade/discussions/17
PRIOR_MODEL = "stabilityai/stable-cascade-prior"
PRIOR_VARIANT = None
PRIOR_CUSTOM_PIPELINE = None
PRIOR_SCHEDULER = None
PRIOR_LORA = None
PRIOR_CONTROLNET = None
PRIOR_DTYPE = "bfloat16"

DECODER_MODEL = "stabilityai/stable-cascade"
DECODER_VARIANT = None
DECODER_CUSTOM_PIPELINE = None
DECODER_SCHEDULER = None
DECODER_LORA = None
DECODER_CONTROLNET = None
DECODER_DTYPE = "float16"

PROMPT = "Anthropomorphic cat dressed as a pilot"
NEGATIVE_PROMPT = None

PRIOR_STEPS = 20
PRIOR_SEED = None
PRIOR_EXTRA_CALL_KWARGS = """{
    "guidance_scale": 4.0
}"""

DECODER_STEPS = 10
DECODER_SEED = None
DECODER_EXTRA_CALL_KWARGS = """{
    "guidance_scale": 0.0,
    "output_type": "pil"
}"""

WARMUPS = 3
BATCH = 1
HEIGHT = 1024
WIDTH = 1024
INPUT_IMAGE = None
PRIOR_CONTROL_IMAGE = None
DECODER_CONTROL_IMAGE = None
OUTPUT_IMAGE = None

import argparse
import importlib
import inspect
import json
import os
import time
from contextlib import nullcontext

import torch
from diffusers.utils import load_image
from PIL import Image, ImageDraw

import oneflow as flow  # usort: skip
from onediffx import compile_pipe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-model", type=str, default=PRIOR_MODEL)
    parser.add_argument("--prior-variant", type=str, default=PRIOR_VARIANT)
    parser.add_argument(
        "--prior-custom-pipeline", type=str, default=PRIOR_CUSTOM_PIPELINE
    )
    parser.add_argument("--prior-scheduler", type=str, default=PRIOR_SCHEDULER)
    parser.add_argument("--prior-lora", type=str, default=PRIOR_LORA)
    parser.add_argument("--prior-controlnet", type=str, default=PRIOR_CONTROLNET)
    parser.add_argument("--prior-dtype", type=str, default=PRIOR_DTYPE)

    parser.add_argument("--decoder-model", type=str, default=DECODER_MODEL)
    parser.add_argument("--decoder-variant", type=str, default=DECODER_VARIANT)
    parser.add_argument(
        "--decoder-custom-pipeline", type=str, default=DECODER_CUSTOM_PIPELINE
    )
    parser.add_argument("--decoder-scheduler", type=str, default=DECODER_SCHEDULER)
    parser.add_argument("--decoder-lora", type=str, default=DECODER_LORA)
    parser.add_argument("--decoder-controlnet", type=str, default=DECODER_CONTROLNET)
    parser.add_argument("--decoder-dtype", type=str, default=DECODER_DTYPE)

    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--negative-prompt", type=str, default=NEGATIVE_PROMPT)

    parser.add_argument("--prior-steps", type=int, default=PRIOR_STEPS)
    parser.add_argument("--prior-seed", type=int, default=PRIOR_SEED)
    parser.add_argument(
        "--prior-extra-call-kwargs", type=str, default=PRIOR_EXTRA_CALL_KWARGS
    )

    parser.add_argument("--decoder-steps", type=int, default=DECODER_STEPS)
    parser.add_argument("--decoder-seed", type=int, default=DECODER_SEED)
    parser.add_argument(
        "--decoder-extra-call-kwargs", type=str, default=DECODER_EXTRA_CALL_KWARGS
    )

    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--input-image", type=str, default=INPUT_IMAGE)
    parser.add_argument("--prior-control-image", type=str, default=PRIOR_CONTROL_IMAGE)
    parser.add_argument(
        "--decoder-control-image", type=str, default=DECODER_CONTROL_IMAGE
    )
    parser.add_argument("--output-image", type=str, default=OUTPUT_IMAGE)
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
    dtype=torch.float16,
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

        pipe = QuantPipeline.from_pretrained(
            pipeline_cls, model_name, torch_dtype=dtype, **extra_kwargs
        )
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
    assert args.input_image is None
    try:
        from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
    except ImportError:
        raise ImportError(
            "Please install `diffusers` from this branch while the PR is WIP: `pip install --upgrade git+https://github.com/kashif/diffusers.git@a3dc21385b7386beb3dab3a9845962ede6765887`"
        )

    prior_pipe = load_pipe(
        StableCascadePriorPipeline,
        args.prior_model,
        variant=args.prior_variant,
        custom_pipeline=args.prior_custom_pipeline,
        scheduler=args.prior_scheduler,
        lora=args.prior_lora,
        controlnet=args.prior_controlnet,
        dtype=getattr(torch, args.prior_dtype),
    )

    patch_oneflow_prior_fp16_overflow = nullcontext
    if prior_pipe.dtype == torch.float16:
        if args.compiler == "oneflow":
            from patch_stable_cascade_of import patch_oneflow_prior_fp16_overflow
        # Dynamic patching would fail with oneflow
        from patch_stable_cascade import patch_prior_fp16_overflow

        prior_pipe.prior = patch_prior_fp16_overflow(prior_pipe.prior)

    decoder_pipe = load_pipe(
        StableCascadeDecoderPipeline,
        args.decoder_model,
        variant=args.decoder_variant,
        custom_pipeline=args.decoder_custom_pipeline,
        scheduler=args.decoder_scheduler,
        lora=args.decoder_lora,
        controlnet=args.decoder_controlnet,
        dtype=getattr(torch, args.decoder_dtype),
    )

    height = args.height
    width = args.width

    if args.compiler == "none":
        pass
    elif args.compiler == "oneflow":
        prior_pipe = compile_pipe(prior_pipe)
        decoder_pipe = compile_pipe(decoder_pipe)
    elif args.compiler in ("compile", "compile-max-autotune"):
        from patch_stable_cascade import patch_torch_compile

        patch_torch_compile()

        mode = "max-autotune" if args.compiler == "compile-max-autotune" else None

        prior_pipe.prior = torch.compile(prior_pipe.prior, mode=mode)
        if hasattr(prior_pipe, "controlnet"):
            prior_pipe.controlnet = torch.compile(prior_pipe.controlnet, mode=mode)

        decoder_pipe.decoder = torch.compile(decoder_pipe.decoder, mode=mode)
        if hasattr(decoder_pipe, "controlnet"):
            decoder_pipe.controlnet = torch.compile(decoder_pipe.controlnet, mode=mode)
    else:
        raise ValueError(f"Unknown compiler: {args.compiler}")

    if args.input_image is None:
        input_image = None
    else:
        input_image = load_image(args.input_image)
        input_image = input_image.resize((width, height), Image.LANCZOS)

    if args.prior_control_image is None:
        if args.prior_controlnet is None:
            prior_control_image = None
        else:
            prior_control_image = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(prior_control_image)
            draw.ellipse(
                (args.width // 4, height // 4, args.width // 4 * 3, height // 4 * 3),
                fill=(255, 255, 255),
            )
            del draw
    else:
        prior_control_image = load_image(args.prior_control_image)
        prior_control_image = prior_control_image.resize((width, height), Image.LANCZOS)

    if args.decoder_control_image is None:
        if args.decoder_controlnet is None:
            decoder_control_image = None
        else:
            decoder_control_image = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(decoder_control_image)
            draw.ellipse(
                (args.width // 4, height // 4, args.width // 4 * 3, height // 4 * 3),
                fill=(255, 255, 255),
            )
            del draw
    else:
        decoder_control_image = load_image(args.decoder_control_image)
        decoder_control_image = decoder_control_image.resize(
            (width, height), Image.LANCZOS
        )

    def get_prior_kwarg_inputs():
        kwarg_inputs = dict(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=height,
            width=width,
            num_inference_steps=args.prior_steps,
            num_images_per_prompt=args.batch,
            generator=None
            if args.prior_seed is None
            else torch.Generator(device="cuda").manual_seed(args.prior_seed),
            **(
                dict()
                if args.prior_extra_call_kwargs is None
                else json.loads(args.prior_extra_call_kwargs)
            ),
        )
        if input_image is not None:
            kwarg_inputs["image"] = input_image
        if prior_control_image is not None:
            if input_image is None:
                kwarg_inputs["image"] = prior_control_image
            else:
                kwarg_inputs["control_image"] = prior_control_image
        return kwarg_inputs

    def get_decoder_kwarg_inputs():
        kwarg_inputs = dict(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.decoder_steps,
            num_images_per_prompt=args.batch,
            generator=None
            if args.decoder_seed is None
            else torch.Generator(device="cuda").manual_seed(args.decoder_seed),
            **(
                dict()
                if args.decoder_extra_call_kwargs is None
                else json.loads(args.decoder_extra_call_kwargs)
            ),
        )
        if input_image is not None:
            kwarg_inputs["image"] = input_image
        if decoder_control_image is not None:
            if input_image is None:
                kwarg_inputs["image"] = decoder_control_image
            else:
                kwarg_inputs["control_image"] = decoder_control_image
        return kwarg_inputs

    # NOTE: Warm it up.
    # The initial calls will trigger compilation and might be very slow.
    # After that, it should be very fast.
    if args.warmups > 0:
        print("Begin warmup")
        for _ in range(args.warmups):
            with patch_oneflow_prior_fp16_overflow():
                prior_output = prior_pipe(**get_prior_kwarg_inputs())
            decoder_pipe(
                image_embeddings=prior_output.image_embeddings.to(
                    dtype=getattr(torch, args.decoder_dtype)
                ),
                **get_decoder_kwarg_inputs(),
            )
        print("End warmup")

    # Let"s see it!
    # Note: Progress bar might work incorrectly due to the async nature of CUDA.
    prior_kwarg_inputs = get_prior_kwarg_inputs()
    prior_iter_profiler = IterationProfiler()
    if "callback_on_step_end" in inspect.signature(prior_pipe).parameters:
        prior_kwarg_inputs[
            "callback_on_step_end"
        ] = prior_iter_profiler.callback_on_step_end
    elif "callback" in inspect.signature(prior_pipe).parameters:
        prior_kwarg_inputs["callback"] = prior_iter_profiler.callback_on_step_end

    decoder_kwarg_inputs = get_decoder_kwarg_inputs()
    decoder_iter_profiler = IterationProfiler()
    if "callback_on_step_end" in inspect.signature(decoder_pipe).parameters:
        decoder_kwarg_inputs[
            "callback_on_step_end"
        ] = decoder_iter_profiler.callback_on_step_end
    elif "callback" in inspect.signature(decoder_pipe).parameters:
        decoder_kwarg_inputs["callback"] = decoder_iter_profiler.callback_on_step_end

    prior_begin = time.time()
    with patch_oneflow_prior_fp16_overflow():
        prior_output = prior_pipe(**prior_kwarg_inputs)
    prior_end = time.time()

    decoder_begin = time.time()
    output_images = decoder_pipe(
        image_embeddings=prior_output.image_embeddings.to(
            dtype=getattr(torch, args.decoder_dtype)
        ),
        **decoder_kwarg_inputs,
    ).images
    decoder_end = time.time()

    print("=======================================")

    print(f"Prior Inference time: {prior_end - prior_begin:.3f}s")
    prior_iter_per_sec = prior_iter_profiler.get_iter_per_sec()
    if prior_iter_per_sec is not None:
        print(f"Prior Iterations per second: {prior_iter_per_sec:.3f}")

    print(f"Decoder Inference time: {decoder_end - decoder_begin:.3f}s")
    decoder_iter_per_sec = decoder_iter_profiler.get_iter_per_sec()
    if decoder_iter_per_sec is not None:
        print(f"Decoder Iterations per second: {decoder_iter_per_sec:.3f}")

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
