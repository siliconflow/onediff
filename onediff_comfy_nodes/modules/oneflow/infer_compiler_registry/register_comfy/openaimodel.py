from abc import abstractmethod

import comfy
import oneflow as th  # 'th' is the way ComfyUI name the torch
import oneflow.nn.functional as F
from einops import rearrange
from onediff.infer_compiler.backends.oneflow.transform import proxy_class, transform_mgr

onediff_comfy = transform_mgr.transform_package("comfy")

ops = onediff_comfy.ops.disable_weight_init
ResBlock = onediff_comfy.ldm.modules.diffusionmodules.openaimodel.ResBlock
checkpoint = onediff_comfy.ldm.modules.diffusionmodules.util.checkpoint


class Upsample(proxy_class(comfy.ldm.modules.diffusionmodules.openaimodel.Upsample)):
    # https://github.com/comfyanonymous/ComfyUI/blob/b0aab1e4ea3dfefe09c4f07de0e5237558097e22/comfy/ldm/modules/diffusionmodules/openaimodel.py#L82
    def forward(self, x, output_shape=None):
        assert x.shape[1] == self.channels
        if output_shape is not None and isinstance(output_shape, th.Tensor):
            if self.dims == 3:
                raise ValueError("output_shape shoud not be Tensor for dims == 3")
            else:
                x = F.interpolate_like(x, like=output_shape, mode="nearest")
        else:
            if self.dims == 3:
                shape = [x.shape[2], x.shape[3] * 2, x.shape[4] * 2]
                if output_shape is not None:
                    shape[1] = output_shape[3]
                    shape[2] = output_shape[4]
            else:
                shape = [x.shape[2] * 2, x.shape[3] * 2]
                if output_shape is not None:
                    shape[0] = output_shape[2]
                    shape[1] = output_shape[3]

            x = F.interpolate(x, size=shape, mode="nearest")

        if self.use_conv:
            x = self.conv(x)
        return x


class UNetModel(proxy_class(comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel)):
    # https://github.com/comfyanonymous/ComfyUI/blob/b0aab1e4ea3dfefe09c4f07de0e5237558097e22/comfy/ldm/modules/diffusionmodules/openaimodel.py#L823
    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        control=None,
        transformer_options={},
        **kwargs
    ):
        timestep_embedding = (
            onediff_comfy.ldm.modules.diffusionmodules.util.timestep_embedding
        )
        forward_timestep_embed = (
            onediff_comfy.ldm.modules.diffusionmodules.openaimodel.forward_timestep_embed
        )
        apply_control = (
            onediff_comfy.ldm.modules.diffusionmodules.openaimodel.apply_control
        )

        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})

        num_video_frames = kwargs.get("num_video_frames", self.default_num_video_frames)
        default_image_only_indicator = getattr(
            self, "default_image_only_indicator", None
        )
        # https://github.com/comfyanonymous/ComfyUI/blob/cb7c3a2921cfc0805be0229b4634e1143d60e6fe/comfy/ldm/modules/diffusionmodules/openaimodel.py#L832
        image_only_indicator = kwargs.get(
            "image_only_indicator", default_image_only_indicator
        )
        time_context = kwargs.get("time_context", None)

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(
            timesteps, self.model_channels, repeat_only=False
        ).to(x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for id, module in enumerate(self.input_blocks):
            transformer_options["block"] = ("input", id)
            h = forward_timestep_embed(
                module,
                h,
                emb,
                context,
                transformer_options,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
            h = apply_control(h, control, "input")
            if "input_block_patch" in transformer_patches:
                patch = transformer_patches["input_block_patch"]
                for p in patch:
                    h = p(h, transformer_options)

            hs.append(h)
            if "input_block_patch_after_skip" in transformer_patches:
                patch = transformer_patches["input_block_patch_after_skip"]
                for p in patch:
                    h = p(h, transformer_options)

        transformer_options["block"] = ("middle", 0)
        h = forward_timestep_embed(
            self.middle_block,
            h,
            emb,
            context,
            transformer_options,
            time_context=time_context,
            num_video_frames=num_video_frames,
            image_only_indicator=image_only_indicator,
        )
        h = apply_control(h, control, "middle")

        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, "output")

            if "output_block_patch" in transformer_patches:
                patch = transformer_patches["output_block_patch"]
                for p in patch:
                    h, hsp = p(h, hsp, transformer_options)

            h = th.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                # output_shape = hs[-1].shape
                output_shape = hs[-1]
            else:
                output_shape = None
            h = forward_timestep_embed(
                module,
                h,
                emb,
                context,
                transformer_options,
                output_shape,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


class VideoResBlock(
    proxy_class(comfy.ldm.modules.diffusionmodules.openaimodel.VideoResBlock)
):
    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = None
        if not self.skip_t_emb:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h)
            if emb_out is not None:
                scale, shift = th.chunk(emb_out, 2, dim=1)
                h *= 1 + scale
                h += shift
            h = out_rest(h)
        else:
            if emb_out is not None:
                if self.exchange_temb_dims:
                    # emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
                    # Rewrite for onediff SVD dynamic shape
                    emb_out = emb_out.permute(0, 2, 1, 3, 4)
                h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

    def forward(
        self,
        x: th.Tensor,
        emb: th.Tensor,
        num_video_frames: int,
        image_only_indicator=None,
    ) -> th.Tensor:
        # Rewrite for onediff SVD dynamic shape
        # x = super().forward(x, emb)
        x = checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

        # x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=num_video_frames)
        batch_frames, _, _, _ = x.shape
        batch_size = batch_frames // num_video_frames
        x_mix = x.unflatten(0, shape=(batch_size, -1)).permute(0, 2, 1, 3, 4)
        # x = rearrange(x, "(b t) c h w -> b c t h w", t=num_video_frames)
        x = x.unflatten(0, shape=(batch_size, -1)).permute(0, 2, 1, 3, 4)

        # x = self.time_stack(
        #     x, rearrange(emb, "(b t) ... -> b t ...", t=num_video_frames)
        # )
        x = self.time_stack(x, emb.unflatten(0, shape=(batch_size, -1)))

        x = self.time_mixer(
            x_spatial=x_mix, x_temporal=x, image_only_indicator=image_only_indicator
        )
        # x = rearrange(x, "b c t h w -> (b t) c h w")
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        return x
