import logging
import os
from pathlib import Path

from onediff.utils.log_utils import ConfigurableLogger

from .system_info import get_system_info, system_info_to_markdown


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Generates an incremented file or directory path if it exists, with optional mkdir; args: path, exist_ok=False,
    sep="", mkdir=False.

    Example: runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


logger = ConfigurableLogger()

out_dir = str(increment_path(Path("results") / "exp", exist_ok=False))
logger.configure_logging(name="benchmarks", level=logging.INFO, log_dir=out_dir)

system_info_str = system_info_to_markdown(get_system_info())
with open(os.path.join(out_dir, "system_info.md"), "w") as fp:
    fp.write(system_info_str)
