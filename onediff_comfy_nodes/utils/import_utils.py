import importlib
import importlib.util

_oneflow_available = importlib.util.find_spec("oneflow") is not None

try:
    import oneflow
except ImportError as e:
    _oneflow_available = False


def is_oneflow_available():
    return _oneflow_available


_onediff_quant_available = importlib.util.find_spec("onediff_quant") is not None
try:
    import onediff_quant
except ImportError as e:
    _onediff_quant_available = False

def is_onediff_quant_available():
    return _onediff_quant_available



_nexfort_available = importlib.util.find_spec("nexfort") is not None
try:
    import nexfort
except ImportError as e:
    _nexfort_available = False

def is_nexfort_available():
    return _nexfort_available



