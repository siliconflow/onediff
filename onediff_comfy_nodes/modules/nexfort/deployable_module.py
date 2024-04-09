import time

import torch


class DeployableModule:
    def __init__(self, original_module: torch.nn.Module, compile_fn):
        super().__init__()
        super().__setattr__("original", original_module)
        super().__setattr__("compiled", None)
        super().__setattr__("compile_fn", compile_fn)

    @property
    def __class__(self):
        return self.original.__class__

    def compile(self):
        if self.compiled is None:
            original_class_name = type(self.original).__qualname__
            begin = time.time()
            print(f'Start compiling  {original_class_name}')
            self.compiled = self.compile_fn(self.original)
            end = time.time()
            print(f'Compilation of {original_class_name} completed in {end - begin:.4f} seconds.')
        return self.compiled

    def to(self, *args, **kwargs):
        return self.compile().to(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.compile()(*args, **kwargs)

    def __getattr__(self, name):
        if name in self.__dict__:
            return super().__getattribute__(name)
        return getattr(self.original, name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            super().__setattr__(name, value)
        else:
            setattr(self.original, name, value)

    def __getitem__(self, key):
        return self.original[key]

    def __setitem__(self, key, value):
        self.original[key] = value

    def __delitem__(self, key):
        del self.original[key]

    def __dir__(self):
        return dir(self.original)

    def __repr__(self):
        return f"DeployableModule({self.original})"