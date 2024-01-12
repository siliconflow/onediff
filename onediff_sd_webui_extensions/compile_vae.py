from modules import shared
from modules.sd_vae_approx import model as get_vae_model, sd_vae_approx_models
from onediff.infer_compiler import oneflow_compile

__all__ = ["VaeCompileCtx"]


compiled_models = {}


class VaeCompileCtx(object):
    def __init__(self, use_graph=True, options={}):
        self._use_graph = use_graph
        self._options = options
        # https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/75336dfc84cae280036bc52a6805eb10d9ae30ba/modules/sd_vae_approx.py#L43
        self._model_name = (
            "vaeapprox-sdxl.pt"
            if getattr(shared.sd_model, "is_sdxl", False)
            else "model.pt"
        )
        self._original_model = get_vae_model()

    def __enter__(self):
        if self._original_model is None:
            return
        global compiled_models
        model = compiled_models.get(self._model_name)
        if model is None:
            model = oneflow_compile(
                self._original_model, use_graph=self._use_graph, options=self._options
            )
            compiled_models[self._model_name] = model
        sd_vae_approx_models[self._model_name] = model

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original_model is not None:
            sd_vae_approx_models[self._model_name] = self._original_model
        return False
