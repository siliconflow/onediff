# import time

# import torch
# from diffusers import StableDiffusion3Pipeline
# from onediffx import compile_pipe, quantize_pipe

# device = torch.device("cuda")
# # torch.set_float32_matmul_precision("high")

# # torch._inductor.config.conv_1x1_as_mm = True
# # torch._inductor.config.coordinate_descent_tuning = True
# # torch._inductor.config.epilogue_fusion = False
# # torch._inductor.config.coordinate_descent_check_all_directions = True

# class SD3Generator:
#     def __init__(self, model, compiler_config=None, quantize_config=None):
#         self.pipe = StableDiffusion3Pipeline.from_pretrained(model, torch_dtype=torch.float16)
#         self.pipe.to(device)

#         if compiler_config:
#             print("compile...")
#             self.pipe = self.compile_pipe(self.pipe, compiler_config)

#         if quantize_config:
#             print("quant...")
#             self.pipe = self.quantize_pipe(self.pipe, quantize_config)

#     def warmup(self, args, warmup_iterations=1):
#         warmup_args = args.copy()
#         warmup_args['num_inference_steps'] = 20

#         warmup_args['generator'] = torch.Generator(device=device).manual_seed(0)

#         print("Starting warmup...")
#         for _ in range(warmup_iterations):
#             self.pipe(**warmup_args)
#         print("Warmup complete.")

#     def generate(self, args):
#         self.warmup(args)

#         seed = args.pop("seed", 1)
#         args["generator"] = torch.Generator(device=device).manual_seed(seed)

#         # Run the model
#         start_time = time.time()
#         images = self.pipe(**args).images
#         end_time = time.time()

#         saved_image = args.get("saved_image", "output.png")
#         images[0].save(saved_image)

#         return images[0], end_time - start_time

#     def compile_pipe(self, pipe, compiler_config):
#         options = compiler_config
#         pipe = compile_pipe(
#             pipe, backend="nexfort", options=options, fuse_qkv_projections=True
#         )
#         return pipe

#     def quantize_pipe(self, pipe, quantize_config):
#         pipe = quantize_pipe(pipe, ignores=[], **quantize_config)
#         return pipe

# args = {
#     "prompt": "product photography, world of Warcraft orc warrior, white background",
#     "num_inference_steps": 20,
#     "height": 1024,
#     "width": 1024,
#     "saved_image": "./sd3.png",
#     "seed": 333
# }

# # Compiler and quantization configurations
# compiler_config = {
#     "mode": "quant:max-optimize:max-autotune:freezing:benchmark:low-precision:cudagraphs",
#     "memory_format": "channels_last"
# }
# quantize_config = {
#     "quant_type": "fp8_e4m3_e4m3_dynamic_per_tensor"
# }

# # Initialize and use the generator
# sd3 = SD3Generator("stabilityai/stable-diffusion-3-medium", compiler_config)
# image, inference_time = sd3.generate(args)
# print(f"Generated image saved to {args['saved_image']} in {inference_time:.2f} seconds.")


import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium", torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(
    prompt="a photo of a cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    height=1024,
    width=1024,
    guidance_scale=7.0,
).images[0]

image.save("sd3_hello_world.png")



# pipe.set_progress_bar_config(disable=True)

# pipe.transformer.to(memory_format=torch.channels_last)
# pipe.vae.to(memory_format=torch.channels_last)


