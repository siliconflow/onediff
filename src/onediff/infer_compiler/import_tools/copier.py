"""Copier for package"""
import os
import sys
import shutil
import importlib
import tempfile
from pathlib import Path
from typing import List, Tuple, Union
from .printer import print_red, print_green
from .copy_utils import copy_files, get_matched_files, convert_funcs_to_classes


class PackageCopier:
    def __init__(
        self, old_pkg: Union[str, Path], prefix="mock_", suffix="", use_temp_dir=False
    ):
        self.old_pkg_name, self.old_pkg_path = self._get_path(old_pkg)
        self.new_pkg_name = prefix + self.old_pkg_name + suffix
        if use_temp_dir:
            self.new_pkg_path = Path(tempfile.gettempdir()) / self.new_pkg_name
        else:
            self.new_pkg_path = self.old_pkg_path.parent / self.new_pkg_name
        assert self.old_pkg_path.exists(), f"{self.old_pkg_path} not exists"
        self.register_call = [
            self.copy_package,
            self.rewrite_func2class,
            self.add_init_files,
            self.rewrite_imports,
            self.test_import,
        ]

    def __repr__(self):
        return (
            f"PackageCopier({self.old_pkg_name} -> {self.new_pkg_name}"
            f"\n{self.old_pkg_path} -> {self.new_pkg_path})"
        )

    def _get_path(self, pkg) -> Tuple[str, Path]:
        try:
            pkg = importlib.import_module(pkg)
            pkg_path = Path(pkg.__file__).parent
            pkg_name = pkg.__name__
            return pkg_name, pkg_path
        except Exception as e:
            pkg_path = Path(pkg)
            if pkg_path.exists():
                pkg_name = pkg_path.name
                return pkg_name, pkg_path
            else:
                raise RuntimeError(f"{pkg} not found") from e

    def copy_package(self):
        src = Path(self.old_pkg_path)
        dest = Path(self.new_pkg_path)
        if src == dest:
            print_red(f"src == dest, do nothing")
            return
        if dest.exists():
            print_red(f"{dest} exists, remove it")
            shutil.rmtree(dest)

        file_list = get_matched_files(src)
        copy_files(src, dest, file_list)

    def add_init_files(self):

        directory = self.new_pkg_path
        if not directory.is_dir():
            raise ValueError(f"{directory} is not a directory")

        def find_directories_with_init(directory: Union[str, Path]) -> List[Path]:
            # Find directories with __init__.py file in them
            directory = Path(directory)
            if not directory.is_dir():
                return []

            result = []
            for path in directory.iterdir():
                if path.is_file():
                    if path.name != "__init__.py" and path.name.endswith(".py"):
                        result.append(path.name.replace(".py", ""))
                elif path.is_dir():
                    if (path / "__init__.py").exists():
                        result.append(path.name)
            return result

        def apply_fn(path: Path) -> None:
            # Apply function to the path
            if path.exists():
                return
            else:
                with open(path, "w") as fp:
                    result = find_directories_with_init(path.parent)
                    if result:
                        fp.write("# This file is created by PackageCopier\n")
                        fp.write("# Add the following lines to your code:\n")
                        for p in result:
                            fp.write(f"from .{p} import *\n")
                        fp.write("# End of the lines\n")

        apply_fn(directory / "__init__.py")

        for path in directory.rglob("*"):
            if path.is_dir() and path.name != "__pycache__":
                apply_fn(path / "__init__.py")

    def rewrite_imports(self):
        for pyfile in self.new_pkg_path.glob("**/*.py"):
            with open(pyfile, "r") as fp:
                content = fp.read()
                content = content.replace(
                    f"{self.old_pkg_name}.", f"{self.new_pkg_name}."
                )
                content = content.replace(
                    f"from {self.old_pkg_name}", f"from {self.new_pkg_name}"
                )
                content = content.replace(
                    f"import {self.old_pkg_name}", f"import {self.new_pkg_name}"
                )
            with open(pyfile, "w") as fp:
                fp.write(content)

    def rewrite_func2class(self):
        for pyfile in self.new_pkg_path.glob("**/*.py"):
            convert_funcs_to_classes(pyfile, pyfile)  # in-place

    def test_import(self):
        sys.path.insert(0, str(self.new_pkg_path.parent))
        importlib.import_module(self.new_pkg_name)
        print_green(f"Test import {self.new_pkg_name} succeed!")

    def get_import_module(self):
        return importlib.import_module(self.new_pkg_name)

    def __call__(self, verbose=False):
        for fn in self.register_call:
            fn()


if __name__ == "__main__":
    copier = PackageCopier("diffusers", prefix="mock_", suffix="")
    copier()
    print(copier.get_import_module())
