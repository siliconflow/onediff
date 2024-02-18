import os
from typing import Any, List, Optional, Union

from .quantize_utils import setup_onediff_quant, load_calibration_and_quantize_pipeline


class QuantPipeline:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *args,
        **kwargs
    ):
        setup_onediff_quant()

        pipe = cls.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        load_calibration_and_quantize_pipeline(pretrained_model_name_or_path, pipe)

        return pipe
