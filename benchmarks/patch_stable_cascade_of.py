import importlib.metadata
from typing import Optional

import oneflow as torch
import oneflow.nn as nn
import oneflow.nn.functional as F

from onediff.infer_compiler.backends.oneflow.transform import transform_mgr
from packaging import version

diffusers_of = transform_mgr.transform_package("diffusers")
StableCascadeUnet_OF_CLS = (
    diffusers_of.pipelines.stable_cascade.modeling_stable_cascade_common.StableCascadeUnet
)

ResBlockStageB = (
    diffusers_of.pipelines.wuerstchen.modeling_wuerstchen_diffnext.ResBlockStageB
)
AttnBlock = diffusers_of.pipelines.wuerstchen.modeling_wuerstchen_common.AttnBlock
TimestepBlock = (
    diffusers_of.pipelines.wuerstchen.modeling_wuerstchen_common.TimestepBlock
)

num_overflow_up_blocks = 1


class StableCascadeUnet_OF(StableCascadeUnet_OF_CLS):
    def of_up_decode(self, level_outputs, r_embed, clip):
        x = level_outputs[0]
        block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers)
        for i, (up_block, upscaler, repmap) in enumerate(block_group):
            if i == len(self.up_blocks) - num_overflow_up_blocks:
                x = x.to(torch.bfloat16)
                r_embed = r_embed.to(torch.bfloat16)
                clip = clip.to(torch.bfloat16)
            for j in range(len(repmap) + 1):
                for k, block in enumerate(up_block):
                    block_class = block.__class__
                    if isinstance(block, ResBlockStageB):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (
                            x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)
                        ):
                            x = torch.nn.functional.interpolate(
                                x.float(),
                                skip.shape[-2:],
                                mode="bilinear",
                                align_corners=True,
                            )
                        if (
                            skip is not None
                            and i >= len(level_outputs) - num_overflow_up_blocks
                        ):
                            skip = skip.to(torch.bfloat16)
                        x = block(x, skip)
                    elif isinstance(block, AttnBlock):
                        x = block(x, clip)
                    elif isinstance(block, TimestepBlock):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if j < len(repmap):
                    x = repmap[j](x)
            x = upscaler(x)
        return x

    def forward(
        self,
        x,
        r,
        clip_text_pooled,
        clip_text=None,
        clip_img=None,
        effnet=None,
        pixels=None,
        sca=None,
        crp=None,
    ):
        if pixels is None:
            pixels = x.new_zeros(x.size(0), 3, 8, 8)

        # Process the conditioning embeddings
        r_embed = self.gen_r_embedding(r)
        for c in self.config.t_conds:
            if c == "sca":
                cond = sca
            elif c == "crp":
                cond = crp
            else:
                cond = None
            t_cond = cond or torch.zeros_like(r)
            r_embed = torch.cat([r_embed, self.gen_r_embedding(t_cond)], dim=1)
        clip = self.gen_c_embeddings(
            clip_txt_pooled=clip_text_pooled, clip_txt=clip_text, clip_img=clip_img
        )

        # Model Blocks
        x = self.embedding(x)
        if hasattr(self, "effnet_mapper") and effnet is not None:
            x = x + self.effnet_mapper(
                nn.functional.interpolate(
                    effnet, size=x.shape[-2:], mode="bilinear", align_corners=True
                )
            )
        if hasattr(self, "pixels_mapper"):
            x = x + nn.functional.interpolate(
                self.pixels_mapper(pixels),
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        level_outputs = self._down_encode(x, r_embed, clip)
        x = self.of_up_decode(level_outputs, r_embed, clip)

        return self.clf(x).to(torch.float16)


from contextlib import contextmanager

# diffusers.pipelines.stable_cascade.modeling_stable_cascade_common.StableCascadeUnet
from diffusers.pipelines.stable_cascade.modeling_stable_cascade_common import (
    StableCascadeUnet,
)

# torch2oflow_class_map.update({StableCascadeUnet: StableCascadeUnetOflow})
from onediff.infer_compiler.backends.oneflow.transform import register


@contextmanager
def patch_oneflow_prior_fp16_overflow():
    torch2oflow_class_map = {StableCascadeUnet: StableCascadeUnet_OF}
    register(torch2oflow_class_map=torch2oflow_class_map)
    yield
    torch2oflow_class_map = {StableCascadeUnet: StableCascadeUnet_OF_CLS}
    register(torch2oflow_class_map=torch2oflow_class_map)
