import comfy
import oneflow as th  # 'th' is the way ComfyUI name the torch
import oneflow.nn.functional as F
from onediff.infer_compiler.transform import proxy_class
from onediff.infer_compiler.transform import transform_mgr

onediff_comfy = transform_mgr.transform_package("comfy")


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
        image_only_indicator = kwargs.get(
            "image_only_indicator", self.default_image_only_indicator
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
