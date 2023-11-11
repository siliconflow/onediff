
"""Use a text prompt to generate an image.
Usage:
    python text_to_image.py  \
        --model_id "runwayml/stable-diffusion-v1-5" \
        --height 1024 \
        --width 1024 

机器 oneflow28-root:     
    * model_id: /share_nfs/hf_models/stable-diffusion-2-1 
    * model_id: /share_nfs/hf_models/stable-diffusion-v1-5

example: 
    # save graph
    $ python text_to_image.py  \
        --model_id "/share_nfs/hf_models/stable-diffusion-2-1" \
        --height 1024 \
        --width 1024 \
        --warmup 3 \
        --save \
        --file "test_graph_2_1" 
    
    # load graph
    $ python text_to_image.py  \
        --model_id "/share_nfs/hf_models/stable-diffusion-2-1" \
        --height 1024 \
        --width 1024 \
        --load \
        --file "test_graph_2_1"
"""
import argparse
from onediff.infer_compiler import oneflow_compile
from onediff import EulerDiscreteScheduler, rewrite_self_attention
from diffusers import StableDiffusionPipeline
import oneflow as flow
import torch
import time

# 计算函数运行时间和内存使用情况
# from:  https://github.com/Oneflow-Inc/diffusers/blob/4e3acbe0c56376ace260afc4883b333499d8fd45/src/onediff/infer_compiler/utils/cost_util.py#L5
from onediff.infer_compiler.utils.cost_util import cost_cnt



def parse_args():
    parser = argparse.ArgumentParser(description="Simple demo of image generation.")
    parser.add_argument(
        "--prompt", type=str, default="a photo of an astronaut riding a horse on mars"
    )
    parser.add_argument(
        "--model_id", type=str, default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)

    ### add save and load graph ###
    parser.add_argument("--save", action=argparse.BooleanOptionalAction)
    parser.add_argument("--load", action=argparse.BooleanOptionalAction)
    parser.add_argument("--file", type=str, required=False, default="unet_compiled")
    ### end ###

    args = parser.parse_args()
    return args

args = parse_args()


# scheduler：专门为 diffusers优化的 EulerDiscreteScheduler  
#  code position:：diffusers/src/onediff/schedulers
scheduler = EulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")
# 从预训练模型中加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    args.model_id,
    scheduler=scheduler,
    use_auth_token=True,
    revision="fp16",
    variant="fp16",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe = pipe.to("cuda")
# 重写 self-attention 模块
# code position: diffusers/src/onediff/optimization/rewrite_self_attention.py
rewrite_self_attention(pipe.unet)
# Compile unet with oneflow
pipe.unet = oneflow_compile(pipe.unet)

# Load graph
if args.load:
    @cost_cnt
    def _load_graph():
        print(f"Loading graph from {args.file}")
        pipe.unet.load_graph(args.file)
    _load_graph()

prompt = args.prompt

""" oneflow.autocast
Note: The following doc was origined by pytorch, see

https://github.com/pytorch/pytorch/blob/master/torch/amp/autocast_mode.py#L19-L179

Instances of :class:`autocast` serve as context managers or decorators that
allow regions of your script to run in mixed precision.
"""
with flow.autocast("cuda"):
    # Warmup
    for _ in range(args.warmup):
        images = pipe(
            prompt, height=args.height, width=args.width, num_inference_steps=args.steps
        ).images

    # Save graph
    if args.save:
        @cost_cnt
        def _save_graph():
            print(f"Saving graph to {args.file}")
            pipe.unet.save_graph(args.file)
        _save_graph()
