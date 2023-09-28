import math
import torch 
import oneflow as flow
import oneflow.nn.functional as F

class FakeCuda:
    @staticmethod
    def current_device():
        return "cuda:0"
    
    @staticmethod
    def mem_get_info(dev):
        return 1024*1024*1024 , 1024*1024*1024

    @staticmethod
    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=None, is_causal=False):
        batch_size, num_heads, hidden_size, head_dim = query.shape

        result = flow._C.fused_multi_head_attention_inference_v2(
            query=query,
            query_layout="BHMK",
            query_head_size=head_dim,
            key=key,
            key_layout="BHMK",
            value=value,
            value_layout="BHMK",
            output_layout="BM(HK)",
            causal=False
        )

        return result

flow.cuda.current_device = FakeCuda.current_device
flow.cuda.mem_get_info = FakeCuda.mem_get_info
flow.nn.functional.scaled_dot_product_attention = FakeCuda.scaled_dot_product_attention
F.scaled_dot_product_attention = FakeCuda.scaled_dot_product_attention







