import torch
from diffusers import  FluxPipeline

# Reference: https://github.com/huggingface/diffusers/pull/9043
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, revision='refs/pr/1')
# pipe.enable_model_cpu_offload()

prompt = "A cat holding a sign that says hello world"
out = pipe(
    prompt=prompt, 
    guidance_scale=0., 
    height=768, 
    width=1360, 
    num_inference_steps=4, 
    max_sequence_length=256,
).images[0]
out.save("image.png")