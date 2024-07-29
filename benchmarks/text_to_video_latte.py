MODEL = "maxin-cn/Latte-1"
CKPT = "t2v_v20240523.pt"
VARIANT = None
CUSTOM_PIPELINE = None
# SAMPLE_METHOD = "DDIM"
BETA_START = 0.0001
BETA_END = 0.02
BREA_SCHEDULE = "linear"
VARIANCE_TYPE = "learned_range"
STEPS = 50
SEED = 25
WARMUPS = 1
BATCH = 1
HEIGHT = 512
WIDTH = 512
VIDEO_LENGTH = 16
FPS = 8
GUIDANCE_SCALE = 7.5
ENABLE_TEMPORAL_ATTENTIONS = "true"
ENABLE_VAE_TEMPORAL_DECODER = "true"
OUTPUT_VIDEO = "output.mp4"

PROMPT = "An epic tornado attacking above aglowing city at night."

EXTRA_CALL_KWARGS = None
ATTENTION_FP16_SCORE_ACCUM_MAX_M = 0

COMPILER_CONFIG = None


import argparse
import importlib
import inspect
import json
import os
import random
import time

import imageio

import torch
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from diffusers.schedulers import DDIMScheduler
from onediffx import compile_pipe
from PIL import Image, ImageDraw
from transformers import T5EncoderModel, T5Tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--ckpt", type=str, default=CKPT)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--save_graph", action="store_true")
    parser.add_argument("--load_graph", action="store_true")
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--custom-pipeline", type=str, default=CUSTOM_PIPELINE)
    # parser.add_argument("--sample-method", type=str, default=SAMPLE_METHOD)
    parser.add_argument("--beta-start", type=float, default=BETA_START)
    parser.add_argument("--beta-end", type=float, default=BETA_END)
    parser.add_argument("--beta-schedule", type=str, default=BREA_SCHEDULE)
    parser.add_argument(
        "--enable_temporal_attentions",
        type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
        default=ENABLE_TEMPORAL_ATTENTIONS,
    )
    parser.add_argument(
        "--enable_vae_temporal_decoder",
        type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
        default=ENABLE_VAE_TEMPORAL_DECODER,
    )
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--variance-type", type=str, default=VARIANCE_TYPE)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--video-length", type=int, default=VIDEO_LENGTH)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--extra-call-kwargs", type=str, default=EXTRA_CALL_KWARGS)
    parser.add_argument("--output-video", type=str, default=OUTPUT_VIDEO)
    parser.add_argument(
        "--compiler",
        type=str,
        default="nexfort",
        choices=["none", "nexfort", "compile"],
    )
    parser.add_argument(
        "--compiler-config",
        type=str,
        default=COMPILER_CONFIG,
    )
    parser.add_argument(
        "--attention-fp16-score-accum-max-m",
        type=int,
        default=ATTENTION_FP16_SCORE_ACCUM_MAX_M,
    )
    return parser.parse_args()


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

    if os.path.exists(args.model):
        model_path = args.model
    else:
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(repo_id=args.model)

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from models.latte_t2v import LatteT2V
    from sample.pipeline_latte import LattePipeline

    transformer_model = LatteT2V.from_pretrained(
        model_path, subfolder="transformer", video_length=args.video_length
    ).to(device, dtype=torch.float16)

    if args.enable_vae_temporal_decoder:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            args.model, subfolder="vae_temporal_decoder", torch_dtype=torch.float16
        ).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(
            args.model, subfolder="vae", torch_dtype=torch.float16
        ).to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        args.model, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    scheduler = DDIMScheduler.from_pretrained(
        model_path,
        subfolder="scheduler",
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        clip_sample=False,
    )

    pipe = LattePipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        transformer=transformer_model,
    ).to(device)

    if args.compiler == "none":
        pass
    elif args.compiler == "nexfort":
        print("Nexfort backend is now active...")
        if args.compiler_config is not None:
            # config with dict
            options = json.loads(args.compiler_config)
        else:
            # config with string
            options = '{"mode": "max-optimize:max-autotune:freezing:benchmark:low-precision",             \
                "memory_format": "channels_last", "options": {"inductor.optimize_linear_epilogue": false, \
                "triton.fuse_attention_allow_fp16_reduction": false}}'
        pipe = compile_pipe(
            pipe, backend="nexfort", options=options, fuse_qkv_projections=True
        )
    elif args.compiler == "compile":
        if hasattr(pipe, "unet"):
            pipe.unet = torch.compile(pipe.unet)
        if hasattr(pipe, "transformer"):
            pipe.transformer = torch.compile(pipe.transformer)
        if hasattr(pipe, "controlnet"):
            pipe.controlnet = torch.compile(pipe.controlnet)
        pipe.vae = torch.compile(pipe.vae)
    else:
        raise ValueError(f"Unknown compiler: {args.compiler}")

    def get_kwarg_inputs():
        kwarg_inputs = dict(
            prompt=args.prompt,
            video_length=args.video_length,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            enable_temporal_attentions=args.enable_temporal_attentions,
            num_images_per_prompt=1,
            mask_feature=True,
            enable_vae_temporal_decoder=args.enable_vae_temporal_decoder,
            **(
                dict()
                if args.extra_call_kwargs is None
                else json.loads(args.extra_call_kwargs)
            ),
        )
        return kwarg_inputs

    if args.warmups > 0:
        print("=======================================")
        print("Begin warmup")
        begin = time.time()
        for _ in range(args.warmups):
            pipe(**get_kwarg_inputs()).video
        end = time.time()
        print("End warmup")
        print(f"Warmup time: {end - begin:.3f}s")

        print("=======================================")

    kwarg_inputs = get_kwarg_inputs()
    iter_profiler = IterationProfiler()
    if "callback_on_step_end" in inspect.signature(pipe).parameters:
        kwarg_inputs["callback_on_step_end"] = iter_profiler.callback_on_step_end
    elif "callback" in inspect.signature(pipe).parameters:
        kwarg_inputs["callback"] = iter_profiler.callback_on_step_end
    torch.manual_seed(args.seed)
    begin = time.time()
    videos = pipe(**kwarg_inputs).video
    end = time.time()

    print("=======================================")
    print(f"Inference time: {end - begin:.3f}s")
    iter_per_sec = iter_profiler.get_iter_per_sec()
    if iter_per_sec is not None:
        print(f"Iterations per second: {iter_per_sec:.3f}")
    cuda_mem_max_used = torch.cuda.max_memory_allocated() / (1024**3)
    cuda_mem_max_reserved = torch.cuda.max_memory_reserved() / (1024**3)
    print(f"Max used CUDA memory : {cuda_mem_max_used:.3f}GiB")
    print(f"Max reserved CUDA memory : {cuda_mem_max_reserved:.3f}GiB")
    print("=======================================")

    if args.output_video is not None:
        # export_to_video(output_frames[0], args.output_video, fps=args.fps)
        try:
            imageio.mimwrite(
                args.output_video, videos[0], fps=8, quality=9
            )  # highest quality is 10, lowest is 0
        except:
            print("Error when saving {}".format(args.prompt))

    else:
        print("Please set `--output-video` to save the output video")


if __name__ == "__main__":
    main()
