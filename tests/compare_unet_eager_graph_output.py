import os

os.environ["ONEFLOW_MLIR_CSE"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS"] = "1"
os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"
os.environ["ONEFLOW_KERENL_CONV_ENABLE_CUTLASS_IMPL"] = "1"
os.environ["ONEFLOW_KERNEL_GLU_ENABLE_DUAL_GEMM_IMPL"] = "1"
os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
os.environ["ONEFLOW_KERENL_FMHA_ENABLE_TRT_FLASH_ATTN_IMPL"] = "1"
# no diff when enable above envs, fp32 & fp16

# os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
# fp16: mean_diff=0.00116
# fp32: mean_diff=0.00015

# os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
# fp16: mean_diff=0.00059
# fp32: mean_diff=0.00015

# os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"
# fp16: mean_diff=0.00185
# fp32: mean_diff=0.00015



import oneflow as flow
flow.mock_torch.enable()

import unittest
import torch
from diffusers import UNet2DConditionModel

class UNet2DConditionModelTests(unittest.TestCase):
    def test_compare_eager_graph_output(self):
        unet_pratrained_model_path = "/home/ldp/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/a304b1ab1b59dd6c3ba9c40705c29c6de4144096/unet"
        data_type = torch.float32
        # data_type = torch.float16
        loading_kwargs = {'torch_dtype': data_type}
        unet = UNet2DConditionModel.from_pretrained(unet_pratrained_model_path, **loading_kwargs).to("cuda")
        unet.eval()

        def dummy_input():
            batch_size = 2
            num_channels = 4
            sizes = (64, 64)

            noise = torch.rand((batch_size, num_channels) + sizes).to("cuda").to(data_type)
            time_step = torch.tensor([10]).to("cuda")
            encoder_hidden_states = torch.rand((batch_size, 77, 768)).to("cuda").to(data_type)

            return {"sample": noise, "timestep": time_step, "encoder_hidden_states": encoder_hidden_states}

        class UNetGraph(torch.nn.Graph):
            def __init__(self, unet):
                super().__init__()
                self.unet = unet
                self.config.enable_cudnn_conv_heuristic_search_algo(False)

            def build(self, latent_model_input, t, text_embeddings):
                text_embeddings = torch._C.amp_white_identity(text_embeddings)
                return self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        unet_graph = UNetGraph(unet)

        inputs_dict = dummy_input()

        with torch.no_grad():
            eager_res = unet(**inputs_dict).sample

            graph_res = unet_graph(inputs_dict["sample"], inputs_dict["timestep"], inputs_dict["encoder_hidden_states"])

            import numpy as np
            out_1 = eager_res.cpu().numpy()
            out_2 = graph_res.cpu().numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            mean_diff = np.mean(np.abs(out_1 - out_2))
            print(f"mean diff: {mean_diff}")
            self.assertLessEqual(mean_diff, 1e-5)