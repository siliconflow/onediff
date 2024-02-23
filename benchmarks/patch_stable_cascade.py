import torch
from diffusers.pipelines.wuerstchen.modeling_wuerstchen_common import (
    AttnBlock,
    TimestepBlock,
    WuerstchenLayerNorm,
)
from diffusers.pipelines.wuerstchen.modeling_wuerstchen_diffnext import ResBlockStageB


def patch_prior_fp16_overflow(prior, num_overflow_up_blocks=1):
    for i in range(len(prior.up_blocks) - num_overflow_up_blocks, len(prior.up_blocks)):
        prior.up_blocks[i].to(torch.bfloat16)
        prior.up_upscalers[i].to(torch.bfloat16)
        prior.up_repeat_mappers[i].to(torch.bfloat16)

    def _up_decode(self, level_outputs, r_embed, clip):
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

    prior._up_decode = _up_decode.__get__(prior)

    prior.clf.to(torch.bfloat16)
    prior_clf_forward = prior.clf.forward

    def new_prior_clf_forward(x):
        return prior_clf_forward(x).to(torch.float16)

    prior.clf.forward = new_prior_clf_forward

    return prior
