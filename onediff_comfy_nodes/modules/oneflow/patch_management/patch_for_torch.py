import torch

original_copy_ = torch.Tensor.copy_

def new_copy_(self, src, *args, **kwargs):
    # print(f'{__file__}.new_copy_ {self.dtype=}')
    if self.dtype == torch.int8 and src.dtype != torch.int8:
        return
    return original_copy_(self, src, *args, **kwargs)

# Replace the original copy_ method with the new one
torch.Tensor.copy_ = new_copy_


