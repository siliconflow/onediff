from typing import Union
from pathlib import Path
import os
import shutil
import fnmatch
from ..utils.log_utils import LOGGER

__all__ = [
    "copy_files",
    "get_matched_files",
]


def get_matched_files(
    root: Union[str, Path], ignore_file=".gitignore", extra_ignore_rules=["*setup.py"]
):
    ignore_rules = []
    ignore_file = Path(root) / ignore_file
    if ignore_file.exists():
        with ignore_file.open("r", encoding="utf-8") as f:
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
            LOGGER.warning(f"{src} does not exist!")


if __name__ == "__main__":
    input_file = "test_fake_torch/_fake_torch.py"  # input file
    output_file = "output.py"  # output file

    print("Done!")
