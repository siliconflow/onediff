from onediff.infer_compiler.backends.oneflow.transform import register
from onediff.utils.import_utils import is_oneflow_available
from onediff_utils import singleton_decorator


@singleton_decorator
def init_oneflow_backend():
    if not is_oneflow_available():
        raise RuntimeError(
            "Backend oneflow for OneDiff is invalid, please make sure you have installed OneFlow"
        )

    from .mock import ldm, sgm, vae

    register(package_names=["ldm"], torch2oflow_class_map=ldm.torch2oflow_class_map)
    register(package_names=["sgm"], torch2oflow_class_map=sgm.torch2oflow_class_map)
    register(package_names=["modules"], torch2oflow_class_map=vae.torch2oflow_class_map)
