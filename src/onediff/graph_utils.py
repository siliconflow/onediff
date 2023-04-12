import os
import oneflow as flow


def get_unet_graph(cache_size, enable_shared, enable_save, *args, **kwargs):
    class UNetGraph(flow.nn.Graph):
        @flow.nn.Graph.with_dynamic_input_shape(
            enable_shared=enable_shared, size=cache_size
        )
        def __init__(self, unet):
            super().__init__(enable_get_runtime_state_dict=enable_save)
            self.unet = unet
            self.config.enable_cudnn_conv_heuristic_search_algo(False)
            self.config.allow_fuse_add_to_output(True)

        def build(self, latent_model_input, t, text_embeddings):
            text_embeddings = flow._C.amp_white_identity(text_embeddings)
            return self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

    return UNetGraph(*args, **kwargs)


def get_vae_graph(cache_size, enable_shared, enable_save, *args, **kwargs):
    class VaeGraph(flow.nn.Graph):
        @flow.nn.Graph.with_dynamic_input_shape(
            enable_shared=enable_shared, size=cache_size
        )
        def __init__(self, vae_post_process) -> None:
            super().__init__(enable_get_runtime_state_dict=enable_save)
            self.vae_post_process = vae_post_process
            self.config.enable_cudnn_conv_heuristic_search_algo(False)
            self.config.allow_fuse_add_to_output(True)

        def build(self, latents):
            return self.vae_post_process(latents)

    return VaeGraph(*args, **kwargs)


class VaePostProcess(flow.nn.Module):
    def __init__(self, vae) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image


class GraphCacheMixin(object):
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

    def __init__(self) -> None:
        self.graph_dict = dict()
        self.cache_size = 10
        self.enable_shared = True
        self.enable_save = True

    def set_graph_compile_cache_size(self, cache_size):
        self.cache_size = cache_size

    def enable_save_graph(self, mode=True):
        self.enable_save = mode

    def enable_graph_share_mem(self, mode=True):
        self.enable_shared = mode

    def save_graph(self, path):
        if self.enable_save:
            for graph_class_name, graph in self.graph_dict.items():
                state_dict = graph.runtime_state_dict()
                flow.save(
                    state_dict, os.path.join(path, graph_class_name),
                )

    def load_graph(self, path, compile_unet: bool = True, compile_vae: bool = True, time_flag: bool = False):
        import time
        # compile vae graph
        vae_graph = None
        if compile_vae:
            vae_post_process = VaePostProcess(self.vae)
            vae_post_process.eval()
            state_dict = flow.load(os.path.join(path, "vae"))
            vae_graph = get_vae_graph(
                cache_size=self.cache_size,
                enable_shared=self.enable_shared,
                enable_save=self.enable_save,
                vae_post_process=vae_post_process,
            )
            flow._oneflow_internal.eager.Sync()
            start = time.time()
            vae_graph.load_runtime_state_dict(state_dict)
            flow._oneflow_internal.eager.Sync()
            if time_flag:
                print(f"Loading time of VAE graph: {time.time() - start:.2f}s")
            self.graph_dict["vae"] = vae_graph

        # compile unet graph
        unet_graph = None
        if compile_unet:
            state_dict = flow.load(os.path.join(path, "unet"))
            unet_graph = get_unet_graph(
                cache_size=self.cache_size,
                enable_shared=self.enable_shared,
                enable_save=self.enable_save,
                unet=self.unet,
            )
            flow._oneflow_internal.eager.Sync()
            start = time.time()
            unet_graph.load_runtime_state_dict(state_dict)
            flow._oneflow_internal.eager.Sync()
            if time_flag:
                print(f"Loading time of UNET graph: {time.time() - start :.2f}s")
            self.graph_dict["unet"] = unet_graph

    def get_graph(self, graph_class, graph):
        if graph_class == "unet":
            if graph_class not in self.graph_dict:
                self.graph_dict[graph_class] = get_unet_graph(
                    cache_size=self.cache_size,
                    enable_shared=self.enable_shared,
                    enable_save=self.enable_save,
                    unet=graph,
                )
            return self.graph_dict[graph_class]
        elif graph_class == "vae":
            if graph_class not in self.graph_dict:
                vae_post_process = VaePostProcess(graph)
                vae_post_process.eval()
                self.graph_dict[graph_class] = get_vae_graph(
                    cache_size=self.cache_size,
                    enable_shared=self.enable_shared,
                    enable_save=self.enable_save,
                    vae_post_process=vae_post_process,
                )
            return self.graph_dict[graph_class]
