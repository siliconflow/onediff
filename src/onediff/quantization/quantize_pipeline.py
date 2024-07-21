import os
from functools import partial
from typing import Any, List, Optional, Union

from onediff_quant import quantize_pipeline, save_quantized

from .quantize_utils import load_calibration_and_quantize_pipeline, setup_onediff_quant


class QuantPipeline:
    @classmethod
    def from_quantized(
        self,
        cls,
        quantized_model_name_or_path: Optional[Union[str, os.PathLike]],
        *args,
        **kwargs
    ):
        """load a quantized model.

        - Example:
          ```python
          from diffusers import AutoPipelineForText2Image
          pipe = QuantPipeline.from_quantized(
              AutoPipelineForText2Image, quantized_model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
          )

          pipe_kwargs = dict(
              prompt=args.prompt,
              height=1024,
              width=1024,
              num_inference_steps=30,
          )
          pipe(**pipe_kwargs)
          ```
        """
        setup_onediff_quant()
        pipe = cls.from_pretrained(quantized_model_name_or_path, *args, **kwargs)
        load_calibration_and_quantize_pipeline(
            os.path.join(str(quantized_model_name_or_path), "calibrate_info.txt"), pipe
        )
        return pipe

    @classmethod
    def from_pretrained(
        self,
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *args,
        **kwargs
    ):
        """load a floating model that to be quantized as int8.

        - Example:
          ```python
          from diffusers import AutoPipelineForText2Image
          pipe = QuantPipeline.from_pretrained(
              AutoPipelineForText2Image, floatting_model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
          )
          pipe.to("cuda")

          pipe_kwargs = dict(
              prompt=args.prompt,
              height=1024,
              width=1024,
              num_inference_steps=30,
          )
          pipe.quantize(**pipe_kwargs,
              conv_compute_density_threshold=900,
              linear_compute_density_threshold=300,
              conv_ssim_threshold=0.985,
              linear_ssim_threshold=0.991,
              save_as_float=False,
              cache_dir=None)

          pipe.save_quantized(args.quantized_model, safe_serialization=True)
          ```
        """
        pipe = cls.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        pipe.quantize = partial(quantize_pipeline, pipe)
        pipe.save_quantized = partial(save_quantized, pipe)
        return pipe

    @classmethod
    def from_single_file(
        self,
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *args,
        **kwargs
    ):
        pipe = cls.from_single_file(pretrained_model_name_or_path, *args, **kwargs)
        pipe.quantize = partial(quantize_pipeline, pipe)
        pipe.save_quantized = partial(save_quantized, pipe)
        return pipe
