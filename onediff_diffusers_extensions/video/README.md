# Accelerate SVD using deepcache algorithm

## How to run DeepCache SVD

```python
import os
import torch

from onediff.infer_compiler import oneflow_compile
from diffusers_extensions.deep_cache import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")

pipe.image_encoder = oneflow_compile(pipe.image_encoder)
pipe.unet = oneflow_compile(pipe.unet)
pipe.vae.decoder = oneflow_compile(pipe.vae.decoder)
pipe.vae.encoder = oneflow_compile(pipe.vae.encoder)

os.environ["ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_SCORE_ACCUMULATION_MAX_M"] = '0'

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)

# Warmup
for i in range(1):
    frames = pipe(
        image,
        num_inference_steps=25,
        decode_chunk_size=5,
        num_frames=25,
        fps=7,
        cache_interval=3,
        cache_branch=0,
        generator=generator,
    ).frames[0]

frames = pipe(
    image,
    num_inference_steps=25,
    decode_chunk_size=5,
    num_frames=25,
    fps=7,
    cache_interval=3,
    cache_branch=0,
    generator=generator,
).frames[0]

export_to_video(frames, "generated.mp4", fps=7)
```


## End-to-End time

- A100

||PyTorch|Torch-Compile|Oneflow|OneDiff-DeepCache|
|:-:|:-:|:-:|:-:|:-:|
|576 x 1024, 25 frames, decode chunk size 5| 51.806s | 43.993s| 32.306s |22.632s|
