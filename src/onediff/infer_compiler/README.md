## Prerequisites
- Python 3.10 is required to have the `torch.compile` work properly

## Major caveats

- in `Attention.forward`, `AttnProcessor` is created every time the function is called
