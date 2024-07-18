""" ComfyUI/comfy/ldm/modules/diffusionmodules/util.py """
import oneflow as torch
import oneflow.nn as nn
import oneflow.nn.functional as F
from einops import rearrange


class AlphaBlender(nn.Module):
    strategies = ["learned", "fixed", "learned_with_images"]

    def get_alpha(self, image_only_indicator: torch.Tensor, device) -> torch.Tensor:
        # skip_time_mix = rearrange(repeat(skip_time_mix, 'b -> (b t) () () ()', t=t), '(b t) 1 ... -> b 1 t ...', t=t)
        if self.merge_strategy == "fixed":
            # make shape compatible
            # alpha = repeat(self.mix_factor, '1 -> b () t  () ()', t=t, b=bs)
            alpha = self.mix_factor.to(device)
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor.to(device))
            # make shape compatible
            # alpha = repeat(alpha, '1 -> s () ()', s = t * bs)
        elif self.merge_strategy == "learned_with_images":
            # assert image_only_indicator is not None, "need image_only_indicator ..."
            if image_only_indicator is None:
                # alpha = rearrange(
                #     torch.sigmoid(self.mix_factor.to(device)), "... -> ... 1"
                # )
                # Rewrite for onediff SVD dynamic shape
                alpha = torch.sigmoid(self.mix_factor.to(device)).unsqueeze(-1)
            else:
                # alpha = torch.where(
                #     image_only_indicator.bool(),
                #     torch.ones(1, 1, device=image_only_indicator.device),
                #     rearrange(torch.sigmoid(self.mix_factor.to(image_only_indicator.device)), "... -> ... 1"),
                # )
                # Rewrite for onediff SVD dynamic shape
                alpha = torch.where(
                    image_only_indicator.bool(),
                    torch.ones(1, 1, device=image_only_indicator.device),
                    torch.sigmoid(
                        self.mix_factor.to(image_only_indicator.device)
                    ).unsqueeze(-1),
                )
            # alpha = rearrange(alpha, self.rearrange_pattern)
            # Rewrite for onediff SVD dynamic shape, only VideoResBlock, rearrange_pattern="b t -> b 1 t 1 1",
            if self.rearrange_pattern == "b t -> b 1 t 1 1":
                alpha.unsqueeze(1).unsqueeze(3).unsqueeze(4)
            elif self.rearrange_pattern == "b t -> (b t) 1 1":
                alpha.flatten().unsqueeze(-1).unsqueeze(-1)
            else:
                alpha = rearrange(alpha, self.rearrange_pattern)
            # make shape compatible
            # alpha = repeat(alpha, '1 -> s () ()', s = t * bs)
        else:
            raise NotImplementedError()
        return alpha

    def forward(
        self,
        x_spatial,
        x_temporal,
        image_only_indicator=None,
    ) -> torch.Tensor:
        alpha = self.get_alpha(image_only_indicator, x_spatial.device)
        x = (
            alpha.to(x_spatial.dtype) * x_spatial
            + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
        )
        return x
