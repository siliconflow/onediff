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


original_pixel_shuffle = torch.nn.functional.pixel_shuffle


def pixel_shuffle(input, upscale_factor):
    # https://blog.csdn.net/ONE_SIX_MIX/article/details/103757856
    # Thanks ChatGPT: https://chat.openai.com/share/1ba80104-406f-4bb8-8292-105f69e0452e

    assert input.dim() >= 3, "Input tensor must have at least 3 dimensions"

    # Separate batch_dims and original C, H, W dimensions
    *batch_dims, channels, height, width = input.shape

    assert (
        channels % (upscale_factor**2) == 0
    ), "Number of channels must be divisible by the square of the upscale factor"

    # Calculate new channels after applying upscale_factor
    new_channels = channels // (upscale_factor**2)

    # Reshape input to (*batch_dims, new_channels, upscale_factor, upscale_factor, height, width)
    reshaped = input.reshape(
        *batch_dims, new_channels, upscale_factor, upscale_factor, height, width
    )

    # Adjust permute to handle dynamic batch dimensions
    permute_dims = list(range(len(batch_dims))) + [
        len(batch_dims),
        len(batch_dims) + 3,
        len(batch_dims) + 1,
        len(batch_dims) + 4,
        len(batch_dims) + 2,
    ]
    permuted = reshaped.permute(*permute_dims)

    # Final reshape to get to the target shape
    output = permuted.reshape(
        *batch_dims, new_channels, height * upscale_factor, width * upscale_factor
    )

    return output


def pixel_unshuffle(input, downscale_factor):
    # Thanks ChatGPT: https://chat.openai.com/share/1ba80104-406f-4bb8-8292-105f69e0452e

    assert input.dim() >= 3, "Input tensor must have at least 3 dimensions"

    # Separate batch_dims and original C, H, W dimensions
    *batch_dims, channels, height, width = input.shape

    # Ensure H and W are divisible by downscale_factor
    assert (
        height % downscale_factor == 0 and width % downscale_factor == 0
    ), "Height and Width must be divisible by the downscale factor"

    # Reshape
    reshaped = input.reshape(
        *batch_dims,
        channels,
        height // downscale_factor,
        downscale_factor,
        width // downscale_factor,
        downscale_factor,
    )

    # Adjust permutation indices for tensors with dimensions > 4
    permute_dims = list(range(len(batch_dims))) + [
        len(batch_dims),
        len(batch_dims) + 2,
        len(batch_dims) + 4,
        len(batch_dims) + 1,
        len(batch_dims) + 3,
    ]
    permuted = reshaped.permute(*permute_dims)

    # Final reshape
    output = permuted.reshape(
        *batch_dims,
        channels * downscale_factor**2,
        height // downscale_factor,
        width // downscale_factor,
    )

    return output


# pixel_shuffle() and pixel_unshuffle() call Tensor.sizes() which is not supported by dynamo.
def patch_torch_compile():
    torch.nn.functional.pixel_shuffle = pixel_shuffle
    torch.nn.functional.pixel_unshuffle = pixel_unshuffle
