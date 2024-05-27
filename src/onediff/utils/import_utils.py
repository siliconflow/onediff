import importlib
import platform

system = platform.system()


def check_module_availability(module_name):
    spec = importlib.util.find_spec(module_name)

    if spec:
        try:
            importlib.import_module(module_name)
        except ImportError:
            return False
    else:
        return False

    return True


_oneflow_available = check_module_availability("oneflow")
_onediff_quant_available = check_module_availability("onediff_quant")
_nexfort_available = check_module_availability("nexfort")

if system != "Linux":
    print(f"Warning: OneFlow is only supported on Linux. Current system: {system}")
    _oneflow_available = False


def is_oneflow_available():
    return _oneflow_available


def is_onediff_quant_available():
    return _onediff_quant_available


def is_nexfort_available():
    return _nexfort_available
