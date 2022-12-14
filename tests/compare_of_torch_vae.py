import numpy as np

import oneflow as flow
import torch

from diffusers import OneFlowAutoencoderKL, AutoencoderKL

pretrained_model_path = "/home/ldp/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/a304b1ab1b59dd6c3ba9c40705c29c6de4144096/vae"

of_type = flow.float32
torch_type = torch.float32
# max diff: 0.003552675247192383
# mean diff: 4.759765215567313e-05

of_type = flow.float16
torch_type = torch.float16
# max diff: 0.013916015625
# mean diff: 0.0001881122589111328

loading_kwargs = {'torch_dtype': of_type}
of_vae = OneFlowAutoencoderKL.from_pretrained(pretrained_model_path, **loading_kwargs)
of_vae = of_vae.to("cuda")
loading_kwargs = {'torch_dtype': torch_type}
torch_vae = AutoencoderKL.from_pretrained(pretrained_model_path, **loading_kwargs)
torch_vae = torch_vae.to("cuda")

np_input = np.random.rand(2, 4, 64, 64)
of_input = flow.tensor(np_input, device="cuda").to(of_type)
torch_input = torch.tensor(np_input, device="cuda").to(torch_type)


with flow.no_grad():
    latents = 1 / 0.18215 * of_input
    of_image = of_vae.decode(latents).sample
    of_image = (of_image / 2 + 0.5).clamp(0, 1)

with torch.no_grad():
    latents = 1 / 0.18215 * torch_input
    torch_image = torch_vae.decode(latents).sample
    torch_image = (torch_image / 2 + 0.5).clamp(0, 1)

out_1 = of_image.cpu().numpy()
out_2 = torch_image.cpu().numpy()
out_1 = out_1[~np.isnan(out_1)]
out_2 = out_2[~np.isnan(out_2)]
max_diff = np.amax(np.abs(out_1 - out_2))
print(f"max diff: {max_diff}")
mean_diff = np.mean(np.abs(out_1 - out_2))
print(f"mean diff: {mean_diff}")
