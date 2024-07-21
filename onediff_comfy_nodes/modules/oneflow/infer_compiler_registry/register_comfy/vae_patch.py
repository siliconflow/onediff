# ComfyUI/comfy/ldm/modules/diffusionmodules/model.py

import oneflow as torch
from onediff.infer_compiler.backends.oneflow.transform import transform_mgr

onediff_comfy = transform_mgr.transform_package("comfy")

AttnBlock_porxy = onediff_comfy.ldm.modules.diffusionmodules.model.AttnBlock


# https://github.com/comfyanonymous/ComfyUI/blob/b4e915e74560bd2c090f9b4ed6b73b0781b7050e/comfy/ldm/modules/diffusionmodules/model.py#L243
class AttnBlock(AttnBlock_porxy):
    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        B, C, _, _ = x.shape
        # compute attention
        q, k, v = map(
            lambda t: t.reshape(B, 1, C, -1).transpose(2, 3).contiguous(),
            (q, k, v),
        )

        _, _, _, head_dim = q.shape
        is_causal = False
        out = torch._C.fused_multi_head_attention_inference_v2(
            query=q,
            query_layout="BHMK",
            query_head_size=head_dim,
            key=k,
            key_layout="BHMK",
            value=v,
            value_layout="BHMK",
            output_layout="BM(HK)",
            causal=is_causal,
        )

        out = out.transpose(1, 2).reshape(B, C, -1)
        out = out.reshape_as(x)

        out = self.proj_out(out)

        return x + out
