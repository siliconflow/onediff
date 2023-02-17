import oneflow as flow
from .oneflow_graph_compile_cache import OneFlowGraphCompileCache


class UNetGraph(flow.nn.Graph):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.config.enable_cudnn_conv_heuristic_search_algo(False)
        self.config.allow_fuse_add_to_output(True)

    def build(self, latent_model_input, t, text_embeddings):
        text_embeddings = flow._C.amp_white_identity(text_embeddings)
        return self.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample


class VaePostProcess(flow.nn.Module):
    def __init__(self, vae) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image


class VaeGraph(flow.nn.Graph):
    def __init__(self, vae_post_process) -> None:
        super().__init__()
        self.vae_post_process = vae_post_process

    def build(self, latents):
        return self.vae_post_process(latents)


class GraphCacheMixin:
    r"""
    Base class for all models.

    [`DiffusionPipeline`] takes care of storing all components (models, schedulers, processors) for diffusion pipelines
    and handles methods for loading, downloading and saving models as well as a few methods common to all pipelines to:

        - move all PyTorch modules to the device of your choice
        - enabling/disabling the progress bar for the denoising iteration

    Class attributes:

        - **config_name** (`str`) -- name of the config file that will store the class and module names of all
          components of the diffusion pipeline.
        - **_optional_components** (List[`str`]) -- list of all components that are optional so they don't have to be
          passed for the pipeline to function (should be overridden by subclasses).
    """
    config_name = "model_index.json"
    _optional_components = []

    def init_graph_compile_cache(self, cache_size, enable_graph_share_mem=False):
        self.graph_compile_cache = OneFlowGraphCompileCache(
            cache_size, enable_graph_share_mem
        )

    def set_graph_compile_cache_size(self, cache_size):
        self.graph_compile_cache.set_cache_size(cache_size)

    def enable_graph_share_mem(self, enabled=True):
        self.graph_compile_cache.enable_share_mem(enabled)

    def enable_save_graph(self, enabled=True):
        self.graph_compile_cache.enable_save_graph(enabled)

    def save_graph(self, path):
        self.graph_compile_cache.save_graph(path)

    def load_graph(self, path, compile_unet: bool = True, compile_vae: bool = True):
        graph_class2init_args = dict()
        # compile vae graph
        vae_graph = None
        if compile_vae:
            vae_post_process = VaePostProcess(self.vae)
            vae_post_process.eval()
            vae_graph_args = (VaeGraph, vae_post_process)
            graph_class2init_args[VaeGraph.__name__] = vae_graph_args

        # compile unet graph
        unet_graph = None
        if compile_unet:
            unet_graph_args = (UNetGraph, self.unet)
            graph_class2init_args[UNetGraph.__name__] = unet_graph_args

        self.graph_compile_cache.load_graph(path, graph_class2init_args)
