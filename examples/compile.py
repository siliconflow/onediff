from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
# pipeline.vae = torch.compile(pipeline.vae, mode="reduce-overhead", fullgraph=True)
pipeline("An image of a squirrel in Picasso style").images[0]
