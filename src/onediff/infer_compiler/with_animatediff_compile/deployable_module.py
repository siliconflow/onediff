import torch


class DeployableModule:
    __attrs = ["_deployed_original_module", "_deployed_compiled_model"]

    def __init__(self, original_module: torch.nn.Module, compiled_model=None):
        assert compiled_model is not None, "Compiled model is None"
        self._deployed_original_module = original_module
        self._deployed_compiled_model = compiled_model

    @property
    def __class__(self):
        return self._deployed_original_module.__class__

    def set_graph_file(self, file_path):
        self._deployed_compiled_model.set_graph_file(file_path)

    def disalbe_graph_file(self):
        self._deployed_compiled_model.disable_graph_file()

    def enable_parameter_update(self):
        updater = self._deployed_compiled_model._parameter_update_controller
        if updater is not None:
            updater.enable_sync()

    def disable_parameter_update(self):
        updater = self._deployed_compiled_model._parameter_update_controller
        if updater is not None:
            updater.disable_sync()

    def get_graph(self):
        return self._deployed_compiled_model.get_graph()

    def load_graph(self, file_path, device=None, run_warmup=True):
        self._deployed_compiled_model.load_graph(file_path, device, run_warmup)

    def save_graph(self, file_path):
        self._deployed_compiled_model.save_graph(file_path)

    def decode(self, *args, **kwargs):
        return self._deployed_compiled_model.decode(*args, **kwargs)

    def to(self, *args, **kwargs):
        return self._deployed_compiled_model.to(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        out = self._deployed_compiled_model(*args, **kwargs)
        return out

    def __getattr__(self, name):
        if name in DeployableModule.__attrs:
            return super().__getattribute__(self, name)
        return getattr(self._deployed_original_module, name)

    def __setattr__(self, name, value):
        if name in DeployableModule.__attrs:
            return super().__setattr__(name, value)
        setattr(self._deployed_original_module, name, value)

    def __getitem__(self, key):
        return self._deployed_original_module[key]

    def __setitem__(self, key, value):
        self._deployed_original_module[key] = value

    def __delitem__(self, key):
        del self._deployed_original_module[key]

    def __dir__(self):
        return dir(self._deployed_original_module)

    def __repr__(self):
        return f"DeployableModule({self._deployed_original_module})"
