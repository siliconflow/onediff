# OneDiff compiler for inference

## With nexfort compiler backend
### Installation
1. Install nexfort: https://github.com/siliconflow/onediff?tab=readme-ov-file#nexfort
2. Install onediff: https://github.com/siliconflow/onediff?tab=readme-ov-file#3-install-onediff

### Usage
```python
from onediff.infer_compiler import compile

# module is the model you want to compile
options = '{"mode": "O3"}'  # mode can be O2 or O3
compiled = compile(module, backend="nexfort", options=options)
```

### Suggested Modes

| Combination | Description |
| - | - |
| `O2` | This is the most suggested combination of compiler modes. This mode requires support for most models, ensuring model accuracy, and supporting dynamic resolution. |
| `O3` | This aims for efficiency. |
