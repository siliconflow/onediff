MODEL = "facebook/DiT-XL-2-512"
SCHEDULER = "DPMSolverMultistepScheduler"
STEPS = 25
WORDS = ["white shark", "umbrella"]
SEED = None
WARMUPS = 3
OUTPUT_IMAGE = None

import os
import importlib
import inspect
import argparse
import time
import json
import torch
from diffusers import DiTPipeline

from onediff.infer_compiler import oneflow_compile
import oneflow as flow


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--scheduler", type=str, default=SCHEDULER)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--words", type=str, action="extend", nargs="+", default=WORDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--output-image", type=str, default=OUTPUT_IMAGE)
    parser.add_argument(
        "--compiler",
        type=str,
        default="oneflow",
        choices=["none", "oneflow", "compile", "compile-max-autotune"],
    )
    return parser.parse_args()


def load_pipe(
    pipeline_cls, model_name, scheduler=None,
):
    extra_kwargs = {}
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
    pipe.to(torch.device("cuda"))
    return pipe


def main():
    args = parse_args()

    pipe = load_pipe(DiTPipeline, args.model, scheduler=args.scheduler,)
    if args.compiler == "none":
        pass
    elif args.compiler == "oneflow":
        pipe.transformer = oneflow_compile(pipe.transformer)
        pipe.vae.decoder = oneflow_compile(pipe.vae.decoder)
    elif args.compiler in ("compile", "compile-max-autotune"):
        mode = "max-autotune" if args.compiler == "compile-max-autotune" else None
        pipe.transformer = torch.compile(pipe.transformer, mode=mode)
        pipe.vae = torch.compile(pipe.vae, mode=mode)
    else:
        raise ValueError(f"Unknown compiler: {args.compiler}")

    class_ids = pipe.get_label_ids(args.words)

    def get_kwarg_inputs():
        kwarg_inputs = dict(
            class_labels=class_ids,
            num_inference_steps=args.steps,
            generator=None
            if args.seed is None
            else torch.Generator(device="cuda").manual_seed(args.seed),
        )
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
    begin = time.time()
    output_images = pipe(**kwarg_inputs).images
    end = time.time()

    print("=======================================")
    print(f"Inference time: {end - begin:.3f}s")
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
