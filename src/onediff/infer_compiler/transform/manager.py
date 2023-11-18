import sys
import shutil
import types
import tempfile
import atexit

from typing import Dict, List, Union
from contextlib import contextmanager
from pathlib import Path
from ..utils.log_utils import set_logging, LOGGER
from ..import_tools.importer import (
    copy_package,
    get_mock_entity_name,
    load_entity_with_mock,
)

__all__ = ["transform_mgr"]


class TransformManager:
    """TransformManager

    __init__ args:
        `debug_mode`: Whether to print debug info.

        `output_dir`: Directory to save run results.
    """

    def __init__(self, debug_mode=False, output_dir: str = "./output"):
        self.debug_mode = debug_mode    
        self._torch_to_oflow_cls_map = {}
        self._torch_to_oflow_packages_list = []
        self._create_output_dir(output_dir)
        self.logger = set_logging(debug_mode=debug_mode, log_dir=output_dir)
        # Create a temp dir to save mock packages.
        self.temp_dir = tempfile.mkdtemp(
            prefix="oneflow_transform_", dir=self.output_dir
        )

    def _create_output_dir(self, output_dir: str):
        """Create a output dir to save run results."""
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    def cleanup(self):
        self.logger.debug(f"Cleaning up temp dir: {self.temp_dir}")
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def load_class_proxies_from_packages(self, package_names: List[Union[Path, str]]):
        self.logger.debug(f"Loading modules: {package_names}")
        for package_name in package_names:
            copy_package(package_name, self.temp_dir)
            self._torch_to_oflow_packages_list.append(package_name)
            self.logger.info(f"Loaded Mock Torch Package: {package_name} successfully")


    def update_class_proxies(self, class_proxy_dict: Dict[str, type], verbose=True):
        """Update `_torch_to_oflow_cls_map` with `class_proxy_dict`.

        example:
            `class_proxy_dict = {"mock_torch.nn.Conv2d": flow.nn.Conv2d}`

        """
        self._torch_to_oflow_cls_map.update(class_proxy_dict)

        debug_message = f"Updated class proxies: {len(class_proxy_dict)=}"
        debug_message += f"\n{class_proxy_dict}\n"
        self.logger.debug(debug_message)




    def transform_cls(self, full_cls_name: str):
        """Transform a class name to a mock class ."""
        mock_full_cls_name = get_mock_entity_name(full_cls_name)

        if mock_full_cls_name in self._torch_to_oflow_cls_map:
            use_value = self._torch_to_oflow_cls_map[mock_full_cls_name]
            return use_value

        use_value = load_entity_with_mock(mock_full_cls_name)
        self._torch_to_oflow_cls_map[mock_full_cls_name] = use_value
        return use_value
        

    def transform_func(self, func: types.FunctionType):
        """Transform a function to a mock function."""
        return load_entity_with_mock(func)


transform_mgr = TransformManager(debug_mode=True)


def handle_exit():
    exc_type, exc_value, traceback = sys.exc_info()
    if exc_type is not None:
        LOGGER.error(f"Exception: {exc_type}, {exc_value}")
        transform_mgr.cleanup()


atexit.register(transform_mgr.cleanup)
atexit.register(handle_exit)
