import os
from typing import Any, List, Optional, Union
from functools import partial

from onediff_quant import quantize_pipeline, save_quantized
from .quantize_utils import setup_onediff_quant, load_calibration_and_quantize_pipeline


class QuantPipeline:
    @classmethod
    def from_quantized(
        self,
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *args,
        **kwargs
    ):
        setup_onediff_quant()
        pipe = cls.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        load_calibration_and_quantize_pipeline(
            os.path.join(str(pretrained_model_name_or_path), "calibrate_info.txt"), pipe
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
        pipe = cls.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        pipe.quantize = partial(quantize_pipeline, pipe)
        pipe.save_quantized = partial(save_quantized, pipe)
        return pipe
