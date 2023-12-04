import torch

_nested_counter = 0

class AutoInplaceAssign:
    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module
    
    def __enter__(self):
        global _nested_counter
        if _nested_counter == 0:
            self.module.apply(module_convert_parameter)
        _nested_counter += 1

    def __exit__(self, exc_type, exc_value, traceback):
        global _nested_counter
        _nested_counter -= 1
        if _nested_counter == 0:
            self.module.apply(module_unconvert_parameter)
            del self.module


class AutoInplaceCopyTensor(torch.Tensor):
    @property
    def data(self):
        return AutoInplaceCopyTensor(self)

    @data.setter
    def data(self, new_tensor):
        if not isinstance(new_tensor, torch.Tensor):
            raise TypeError
        self.copy_(new_tensor)

class AutoInplaceCopyParameter(torch.nn.Parameter):
    @property
    def data(self):
        return AutoInplaceCopyTensor(super(AutoInplaceCopyParameter, self).data)

    @data.setter
    def data(self, new_tensor):
        if not isinstance(new_tensor, torch.Tensor):
            raise TypeError
        self.data.copy_(new_tensor)


def module_convert_parameter(module: torch.nn.Module):
    for k, v in module.__dict__.items():
        if isinstance(v, torch.Tensor):
            module.__dict__[k] = AutoInplaceCopyTensor(v)
        elif isinstance(v, torch.nn.Parameter):
            module.__dict__[k] = AutoInplaceCopyParameter(v)
    for k, param in module._parameters.items():
        if isinstance(param, (AutoInplaceCopyParameter, AutoInplaceCopyTensor)):
            continue
        if param is not None:
            module._parameters[k] = AutoInplaceCopyParameter(param)
    for k, buffer in module._buffers.items():
        if isinstance(param, (AutoInplaceCopyParameter, AutoInplaceCopyTensor)):
            continue
        if buffer is not None:
            module._buffers[k] = AutoInplaceCopyTensor(buffer)

def module_unconvert_parameter(module: torch.nn.Module):
    for k, v in module.__dict__.items():
        if isinstance(v, AutoInplaceCopyTensor):
            module.__dict__[k] = torch.Tensor(v)
        elif isinstance(v, torch.nn.Parameter):
            module.__dict__[k] = torch.nn.Parameter(torch.Tensor(v.data))
    for k, param in module._parameters.items():
        if not isinstance(param, (AutoInplaceCopyParameter, AutoInplaceCopyTensor)):
            continue
        if param is not None:
            module._parameters[k] = torch.nn.Parameter(torch.Tensor(param.data))
    for k, buffer in module._buffers.items():
        if not isinstance(param, (AutoInplaceCopyParameter, AutoInplaceCopyTensor)):
            continue
        if buffer is not None:
            module._buffers[k] = torch.Tensor(buffer)


if __name__ == "__main__":
    class EagerModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(3, 3)
            self.linear2 = torch.nn.Linear(3, 3)

        def forward(self, x):
            return self.linear2(self.linear1(x))

    eager = EagerModule()
    dptr1 = eager.linear1.weight.data.data_ptr()
    dptr2 = eager.linear2.weight.data.data_ptr()

    with AutoInplaceAssign(eager):
        eager.linear1.weight.data = torch.randn(3, 3)
        eager.linear2.weight.data = torch.randn(3, 3)

    assert dptr1 == eager.linear1.weight.data.data_ptr()
    assert dptr2 == eager.linear2.weight.data.data_ptr()
