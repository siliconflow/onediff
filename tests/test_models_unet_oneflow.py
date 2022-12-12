# coding=utf-8
# Copyright 2022 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

os.environ["ONEFLOW_MLIR_CSE"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"

# above env, no diff when set to 1, fp32 & fp16

# os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
# fp16 max_diff=0.001953125, fp32 max_diff=0.000817

# os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"

# os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS"] = "1"
# os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"

# os.environ["ONEFLOW_KERENL_CONV_ENABLE_CUTLASS_IMPL"] = "1"
# os.environ["ONEFLOW_KERENL_FMHA_ENABLE_TRT_FLASH_ATTN_IMPL"] = "1"
# os.environ["ONEFLOW_KERNEL_GLU_ENABLE_DUAL_GEMM_IMPL"] = "1"

# os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
# os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"

import math
import unittest

import oneflow as torch

from diffusers import OneFlowUNet2DConditionModel
from diffusers.testing_oneflow_utils import floats_tensor, slow, torch_device

from tests.test_modeling_common_oneflow import ModelTesterMixin


class UNet2DConditionModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = OneFlowUNet2DConditionModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)
        encoder_hidden_states = floats_tensor((batch_size, 4, 32)).to(torch_device)

        return {"sample": noise, "timestep": time_step, "encoder_hidden_states": encoder_hidden_states}

    @property
    def input_shape(self):
        return (4, 32, 32)

    @property
    def output_shape(self):
        return (4, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": (32, 64),
            "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
            "cross_attention_dim": 32,
            "attention_head_dim": 8,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 2,
            "sample_size": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_compare_eager_graph_output(self):
        unet_pratrained_model_path = "/home/ldp/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/a304b1ab1b59dd6c3ba9c40705c29c6de4144096/unet"
        data_type = torch.float32
        # data_type = torch.float16
        loading_kwargs = {'torch_dtype': data_type}
        unet = OneFlowUNet2DConditionModel.from_pretrained(unet_pratrained_model_path, **loading_kwargs).to(torch_device)
        unet.eval()

        def dummy_input():
            batch_size = 2
            num_channels = 4
            sizes = (64, 64)

            noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device).to(data_type)
            time_step = torch.tensor([10]).to(torch_device)
            encoder_hidden_states = floats_tensor((batch_size, 77, 768)).to(torch_device).to(data_type)

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
            max_diff = np.amax(np.abs(out_1 - out_2))
            print(f"max diff: {max_diff}")
            self.assertLessEqual(max_diff, 1e-5)






        










