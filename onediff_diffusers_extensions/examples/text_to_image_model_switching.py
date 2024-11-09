"""
python3 onediff_diffusers_extensions/examples/text_to_image_model_switching.py \
    --models \
        runwayml/stable-diffusion-v1-5 \
        /data/home/wangyi/models/base/AWPainting_v1.2.safetensors \
        /data/home/wangyi/models/base/Deliberate_v2.safetensors \
        /data/home/wangyi/models/base/liblib_huanshiyihua_v1.0.safetensors \
        /data/home/wangyi/models/base/realisticVisionV6.0.safetensors \
"""

import argparse
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from onediffx import compile_pipe

from diffusers import StableDiffusionPipeline
from onediff.utils.import_utils import is_nexfort_available, is_oneflow_available

USE_ONEFLOW = is_oneflow_available()
USE_NEXFORT = is_nexfort_available()
IMAGES = defaultdict(OrderedDict)


def load_pipe_weights(compiled_pipe, new_pipe):
    for component_name in compiled_pipe.components:
        compiled_comp = getattr(compiled_pipe, component_name)
        if compiled_comp is None:
            continue
        new_comp = getattr(new_pipe, component_name)
        if hasattr(compiled_comp, "load_state_dict"):
            compiled_comp.load_state_dict(new_comp.state_dict())


def load_pipe(model_name_or_path: str):
    if model_name_or_path.endswith(".safetensors"):
        return StableDiffusionPipeline.from_single_file(
            model_name_or_path,
            variant="fp16",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        return StableDiffusionPipeline.from_pretrained(
            model_name_or_path,
            variant="fp16",
            torch_dtype=torch.float16,
            safety_checker=None,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", type=str, default="a cat"
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=[],
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    cmd_args = parser.parse_args()
    return cmd_args


args = parse_args()

IMAGES = defaultdict(OrderedDict)

# ---------- torch backend ----------
pipe = load_pipe(args.models[0])
pipe = pipe.to("cuda")
for model in args.models:
    print(f"using model: {model}")
    new_pipe = load_pipe(model)
    load_pipe_weights(pipe, new_pipe)
    torch.manual_seed(args.seed)
    image = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
    ).images[0]

    image.save(f"torch-{args.prompt}-of-model-{Path(model).stem}.png")
    IMAGES["torch"][Path(model).stem] = image


# ---------- oneflow backend ----------
if USE_ONEFLOW:
    print("using oneflow backend")
    pipe = load_pipe(args.models[0])
    pipe = pipe.to("cuda")
    pipe = compile_pipe(pipe, backend="oneflow")
    for _ in range(args.warmup):
        torch.manual_seed(args.seed)
        images = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
        ).images

    for model in args.models:
        print(f"using model: {model}")
        new_pipe = load_pipe(model)
        load_pipe_weights(pipe, new_pipe)
        torch.manual_seed(args.seed)
        image = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
        ).images[0]

        image.save(f"oneflow-{args.prompt}-of-model-{Path(model).stem}.png")
        IMAGES["oneflow"][Path(model).stem] = image

# ---------- nexfort backend ----------
if USE_NEXFORT:
    print("using nexfort backend")
    del pipe
    torch.cuda.empty_cache()
    if USE_ONEFLOW:
        import oneflow as flow
        flow.cuda.empty_cache()
    nexfort_options = {
        "mode": "cudagraphs:benchmark:max-autotune:low-precision:cache-all",
        "memory_format": "channels_last",
        "options": {
            "inductor.optimize_linear_epilogue": False,
            "overrides.conv_benchmark": True,
            "overrides.matmul_allow_tf32": True,
        },
    }
    pipe = load_pipe(args.models[0])
    pipe = pipe.to("cuda")
    pipe = compile_pipe(pipe, backend="nexfort", options=nexfort_options)

    for _ in range(args.warmup):
        torch.manual_seed(args.seed)
        images = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
        ).images

    for model in args.models:
        print(f"using model: {model}")
        new_pipe = load_pipe(model)
        load_pipe_weights(pipe, new_pipe)
        torch.manual_seed(args.seed)
        image = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
        ).images[0]

        image.save(f"nexfort-{args.prompt}-of-model-{Path(model).stem}.png")
        IMAGES["nexfort"][Path(model).stem] = image


fig, axs = plt.subplots(len(args.models), 3, figsize=(10, 10))
for i, (backend, images) in enumerate(IMAGES.items()):
    for j, (_, image) in enumerate(images.items()):
        axs[j, i].axis("off")
        axs[j, i].imshow(image)

column_titles = ["torch", "OneDiff (oneflow)", "OneDiff (nexfort)"]
for col in range(3):
    axs[0, col].set_title(column_titles[col])

plt.tight_layout(rect=[0.1, 0, 1, 1])
plt.savefig("onediff_model_switching.png")
