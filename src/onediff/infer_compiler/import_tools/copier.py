"""Copier for package"""
import os
import sys
import shutil
import importlib
import tempfile
import fnmatch
from pathlib import Path
from typing import List, Tuple, Union
from .printer import print_red, print_green


def get_matched_files(
    root: Union[str, Path], ignore_file=".gitignore", extra_ignore_rules=["*setup.py"]
):
    ignore_rules = []
    ignore_file = Path(root) / ignore_file
    if ignore_file.exists():
        with open(ignore_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ignore_rules.append(line)

    ignore_rules.extend(extra_ignore_rules)
    matches = []
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            filepath = os.path.relpath(filepath, root)
            if len(ignore_rules) == 0:
                matches.append(filepath)
                continue
            is_match = any(fnmatch.fnmatch(filepath, rule) for rule in ignore_rules)
            if not is_match:
                matches.append(filepath)

    return matches


def copy_files(src_dir, dst_dir, filelist):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    for file in filelist:
        src = src_dir / file
        dst = dst_dir / file

        if not dst.parent.exists():
            dst.parent.mkdir(parents=True)

        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"{src} does not exist!")


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
            self.add_init_files,
            self.rewrite_imports,
        ]

    def __enter__(self):
        # Copy the package to a new place
        self.__call__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove the new package after exit
        shutil.rmtree(self.new_pkg_path)
        if exc_tb:
            print(f"{exc_type=} {exc_val=} {exc_tb=}")

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

        def apply_fn(path: Path):
            # Apply function to the path
            if path.exists():
                return
            else:
                with open(path, "w", encoding="utf-8") as fp:
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
            with open(pyfile, "r", encoding="utf-8") as fp:
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
            with open(pyfile, "w", encoding="utf-8") as fp:
                fp.write(content)

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
