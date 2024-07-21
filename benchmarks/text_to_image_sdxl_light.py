MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
REPO = "ByteDance/SDXL-Lightning"
CPKT = "sdxl_lightning_4step_unet.safetensors"
VARIANT = None
CUSTOM_PIPELINE = None
CONTROLNET = None
PROMPT = "A girl smiling"
NEGATIVE_PROMPT = None
SEED = None
WARMUPS = 3
BATCH = 1
HEIGHT = 1024
WIDTH = 1024
OUTPUT_IMAGE = None
EXTRA_CALL_KWARGS = None

import argparse
import importlib
import inspect
import json
import os
import time

import torch
from diffusers.utils import load_image
from PIL import Image, ImageDraw

import oneflow as flow  # usort: skip
from huggingface_hub import hf_hub_download
from onediffx import compile_pipe
from safetensors.torch import load_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--repo", type=str, default=REPO)
    parser.add_argument("--cpkt", type=str, default=CPKT)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--custom-pipeline", type=str, default=CUSTOM_PIPELINE)
    parser.add_argument("--controlnet", type=str, default=CONTROLNET)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--negative-prompt", type=str, default=NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--extra-call-kwargs", type=str, default=EXTRA_CALL_KWARGS)
    parser.add_argument("--output-image", type=str, default=OUTPUT_IMAGE)
    parser.add_argument(
        "--compiler",
        type=str,
        default="oneflow",
        choices=["none", "oneflow", "compile", "compile-max-autotune"],
    )
    return parser.parse_args()


def load_and_compile_pipe(
    model_name,
    repo_name,
    cpkt_name,
    compile_type,
    variant=None,
    custom_pipeline=None,
    controlnet=None,
):
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline

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

        raise TypeError("Quantizatble SDXL-LIGHT is not supported!")
        # pipe = QuantPipeline.from_quantized(
        #     pipeline_cls, model_name, torch_dtype=torch.float16, **extra_kwargs
        # )
    else:
        is_lora_cpkt = "lora" in cpkt_name
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name, torch_dtype=torch.float16, **extra_kwargs
        )

        if is_lora_cpkt:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_name, torch_dtype=torch.float16, **extra_kwargs
            ).to("cuda")
            if os.path.isfile(os.path.join(repo_name, cpkt_name)):
                pipe.load_lora_weights(os.path.join(repo_name, cpkt_name))
            else:
                pipe.load_lora_weights(hf_hub_download(repo_name, cpkt_name))
            pipe.fuse_lora()
        else:
            from diffusers import UNet2DConditionModel

            unet = UNet2DConditionModel.from_config(model_name, subfolder="unet").to(
                "cuda", torch.float16
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
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_name, unet=unet, torch_dtype=torch.float16, **extra_kwargs
            ).to("cuda")

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.safety_checker = None
    pipe.to(torch.device("cuda"))

    if compile_type == "none":
        pass
    elif compile_type == "oneflow":
        pipe = compile_pipe(pipe)
    elif compile_type in ("compile", "compile-max-autotune"):
        mode = "max-autotune" if compile_type == "compile-max-autotune" else None
        pipe.unet = torch.compile(pipe.unet, mode=mode)
        if hasattr(pipe, "controlnet"):
            pipe.controlnet = torch.compile(pipe.controlnet, mode=mode)
        pipe.vae = torch.compile(pipe.vae, mode=mode)
    else:
        raise ValueError(f"Unknown compiler: {compile_type}")

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

    pipe = load_and_compile_pipe(
        args.model,
        args.repo,
        args.cpkt,
        args.compiler,
        variant=args.variant,
        custom_pipeline=args.custom_pipeline,
        controlnet=args.controlnet,
    )

    height = args.height
    width = args.width
    height = args.height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = args.width or pipe.unet.config.sample_size * pipe.vae_scale_factor

    n_steps = int(args.cpkt[len("sdxl_lightning_") : len("sdxl_lightning_") + 1])

    def get_kwarg_inputs():
        kwarg_inputs = dict(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=height,
            width=width,
            num_inference_steps=n_steps,
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
    iter_profiler = None
    if "callback_on_step_end" in inspect.signature(pipe).parameters:
        iter_profiler = IterationProfiler()
        kwarg_inputs["callback_on_step_end"] = iter_profiler.callback_on_step_end
    elif "callback" in inspect.signature(pipe).parameters:
        iter_profiler = IterationProfiler()
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
