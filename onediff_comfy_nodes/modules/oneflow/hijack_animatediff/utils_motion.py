# ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/animatediff/utils_motion.py
import oneflow as torch
from onediff.infer_compiler.backends.oneflow.transform import register

from ._config import animatediff_of, animatediff_pt

# ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/animatediff/utils_motion.py
CrossAttentionMM_OF_CLS = animatediff_of.animatediff.utils_motion.CrossAttentionMM
CrossAttentionMM_PT_CLS = animatediff_pt.animatediff.utils_motion.CrossAttentionMM


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d


def attention_pytorch(q, k, v, heads, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    # q, k, v = map(
    #     lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
    #     (q, k, v),
    # )

    # out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    # out = (
    #     out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    # )
    assert mask is None
    out = torch._C.fused_multi_head_attention_inference_v2(
        query=q,
        query_layout="BM(HK)",
        query_head_size=dim_head,
        key=k,
        key_layout="BM(HK)",
        value=v,
        value_layout="BM(HK)",
        output_layout="BM(HK)",
        causal=False,
    )

    return out


class CrossAttentionMM_OF(CrossAttentionMM_OF_CLS):
    def forward(self, x, context=None, value=None, mask=None, scale_mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        # apply custom scale by multiplying k by scale factor
        if self.scale is not None:
            k *= self.scale

        # apply scale mask, if present
        if scale_mask is not None:
            k *= scale_mask

        # out = optimized_attention_mm(q, k, v, self.heads, mask)
        out = attention_pytorch(q, k, v, self.heads, mask)

        return self.to_out(out)


register(torch2oflow_class_map={CrossAttentionMM_PT_CLS: CrossAttentionMM_OF})
