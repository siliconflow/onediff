from onediff_utils import singleton_decorator

from onediff.infer_compiler.backends.oneflow.transform import register

from .mock import ldm, sgm


@singleton_decorator
def init_oneflow_backend():
    register(package_names=["ldm"], torch2oflow_class_map=ldm.torch2oflow_class_map)
    register(package_names=["sgm"], torch2oflow_class_map=sgm.torch2oflow_class_map)
