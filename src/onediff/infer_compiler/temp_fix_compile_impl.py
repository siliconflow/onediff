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
    

    # torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    @staticmethod
    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=None, is_causal=False)-> flow.Tensor:
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
            
        if attn_mask is not None:
            if attn_mask.dtype == flow.bool:
                # attn_mask = flow.cast(attn_mask, flow.float32)
                # new_attn_mask = flow.zeros_like(attn_mask)
                new_attn_mask =flow.empty(attn_mask.shape, dtype=query.dtype, device=query.device)
                mask = flow.logical_not(attn_mask)
                # -std::numeric_limits<double>::infinity()
                new_attn_mask.masked_fill_(mask, float('-inf'))   
                attn_mask = new_attn_mask 
        
        scores = flow.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        
        if attn_mask is not None:
            scores.add_(attn_mask)
        
        p_attn = flow.nn.functional.softmax(scores, dim = -1)

        return flow.matmul(p_attn, value)
        

flow.cuda.current_device = FakeCuda.current_device
flow.cuda.mem_get_info = FakeCuda.mem_get_info
flow.nn.functional.scaled_dot_product_attention = FakeCuda.scaled_dot_product_attention
F.scaled_dot_product_attention = FakeCuda.scaled_dot_product_attention







