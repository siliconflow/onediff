import os

from modules import shared


class SD21CompileCtx(object):
    """to avoid results for NaN when the model is v2-1_768-ema-pruned"""

    _var_name = "ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_ACCUMULATION"

    def __enter__(self):
        self._original = os.getenv(self._var_name)
        if shared.opts.sd_model_checkpoint.startswith("v2-1"):
            os.environ[self._var_name] = "0"

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original is not None:
            os.environ[self._var_name] = self._original
        return False
