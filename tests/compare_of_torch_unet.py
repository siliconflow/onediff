import numpy as np

import oneflow as flow
import torch

from diffusers import OneFlowUNet2DConditionModel, UNet2DConditionModel

pretrained_model_path = "/home/ldp/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/a304b1ab1b59dd6c3ba9c40705c29c6de4144096/unet"

of_type = flow.float32
torch_type = torch.float32
# time step 1:
# max diff: 0.0015354156494140625
# mean diff: 0.00029023358365520835
# time step 50:
# max diff: 0.3434886932373047
# mean diff: 0.001315166475251317

# of_type = flow.float16
# torch_type = torch.float16
# time step 1:
# max diff: 0.001953125
# mean diff: 0.0003502368927001953
# time step 50:
# max diff: 0.0625
# mean diff: 0.0011386871337890625

time_steps = 50

loading_kwargs = {'torch_dtype': of_type}
of_unet = OneFlowUNet2DConditionModel.from_pretrained(pretrained_model_path, **loading_kwargs)
of_unet = of_unet.to("cuda")
loading_kwargs = {'torch_dtype': torch_type}
torch_unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, **loading_kwargs)
torch_unet = torch_unet.to("cuda")

noise_np = np.random.rand(2, 4, 64, 64)
encoder_hidden_states_np = np.random.rand(2, 77, 768)

with flow.no_grad():
    of_res = flow.tensor(noise_np).to("cuda").to(of_type)
    encoder_hidden_states = flow.tensor(encoder_hidden_states_np).to("cuda").to(of_type)
    for t in range(time_steps):
        t_of = flow.tensor([t]).to("cuda")
        of_res = of_unet(of_res, t_of, encoder_hidden_states).sample

with torch.no_grad():
    torch_res = torch.tensor(noise_np).to("cuda").to(torch_type)
    encoder_hidden_states_torch = torch.tensor(encoder_hidden_states_np).to("cuda").to(torch_type)
    for t in range(time_steps):
        t_torch = torch.tensor([t]).to("cuda")
        torch_res = torch_unet(torch_res, t_torch, encoder_hidden_states_torch).sample

out_1 = of_res.cpu().numpy()
out_2 = torch_res.cpu().numpy()
out_1 = out_1[~np.isnan(out_1)]
out_2 = out_2[~np.isnan(out_2)]
max_diff = np.amax(np.abs(out_1 - out_2))
print(f"max diff: {max_diff}")
mean_diff = np.mean(np.abs(out_1 - out_2))
print(f"mean diff: {mean_diff}")
