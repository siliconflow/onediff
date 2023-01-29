import oneflow as torch
import time
import os
import shutil

from diffusers import (
    OneFlowStableDiffusionPipeline as StableDiffusionPipeline,
    OneFlowEulerDiscreteScheduler as EulerDiscreteScheduler,
)
from diffusers import utils

model_id = "stabilityai/stable-diffusion-2"
_graph_save_file = "./test_sd_save_graph"
_sch_file_path = "./test_sd_sch"
_pipe_file_path = "./test_sd_pipe"

_online_mode = True
_pipe_from_file = True

total_start_t = time.time()
start_t = time.time()
@utils.cost_cnt
def get_pipe():
    if _pipe_from_file:
        scheduler = EulerDiscreteScheduler.from_pretrained(_sch_file_path, subfolder="scheduler")
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            _pipe_file_path, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
            )
    else:
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
            )
    torch._oneflow_internal.eager.Sync()
    return scheduler, sd_pipe
sch, pipe = get_pipe()

@utils.cost_cnt
def pipe_to_cuda():
    cu_pipe = pipe.to("cuda")
    torch._oneflow_internal.eager.Sync()
    return cu_pipe
pipe = pipe_to_cuda()

@utils.cost_cnt
def config_graph():
    pipe.set_graph_compile_cache_size(9)
    pipe.enable_graph_share_mem()
    torch._oneflow_internal.eager.Sync()
config_graph()

if not _online_mode:
    pipe.enable_save_graph()
else:
    @utils.cost_cnt
    def load_graph():
        pipe.enable_load_graph()
        assert (os.path.exists(_graph_save_file) and os.path.isdir(_graph_save_file))
        pipe.load_graph(_graph_save_file, compile_unet=True, compile_vae=False)
        torch._oneflow_internal.eager.Sync()
    load_graph()
end_t = time.time()
print("sd init time ", end_t - start_t, 's.')

@utils.cost_cnt
def text_to_image(prompt, image_size, num_images_per_prompt=1, prefix=""):
    if isinstance(image_size, int):
        image_height = image_size
        image_weight = image_size
    elif isinstance(image_size, (tuple, list)):
        assert len(image_size) == 2
        image_height, image_weight = image_size
    else:
        raise ValueError(f"invalie image_size {image_size}")

    images = pipe(
        prompt,
        height=image_height,
        width=image_weight,
        compile_unet=True,
        compile_vae=False,
        num_images_per_prompt=num_images_per_prompt,
    ).images
    for i, image in enumerate(images):
        image.save(f"{prefix}{prompt}_{image_height}x{image_weight}_{i}.png")


prompt = "a photo of an astronaut riding a horse on mars"

sizes = [1024, 896, 768]
# sizes = [768, 896, 1024]
for i in sizes:
    for j in sizes:
        text_to_image(prompt, (i, j), prefix=f"{0}-")
        # for n in range(3):
        #     text_to_image(prompt, (i, j), prefix=f"{n}-")
torch._oneflow_internal.eager.Sync()
total_end_t = time.time()
print("st init and run time ", total_end_t - total_start_t, 's.')

@utils.cost_cnt
def save_pipe_sch():
    pipe.save_pretrained(_pipe_file_path)
    sch.save_pretrained(_sch_file_path)

@utils.cost_cnt
def save_graph():
    if os.path.exists(_graph_save_file) and os.path.isdir(_graph_save_file):
        shutil.rmtree(_graph_save_file)
        os.makedirs(_graph_save_file)

    pipe.save_graph(_graph_save_file)

if not _online_mode:
    save_pipe_sch()
    save_graph()
