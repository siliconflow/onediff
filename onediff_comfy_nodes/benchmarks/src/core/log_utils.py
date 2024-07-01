import logging
import os
from pathlib import Path

from onediff.utils.log_utils import ConfigurableLogger


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

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


logger = ConfigurableLogger()


def setup_logging(output_dir="results", exp_name="exp"):
    """
    Set up logging for an experiment.

    Args:
        output_dir (str): The base directory where experiment results will be stored.
        exp_name (str): The name of the experiment.

    Returns:
        tuple: A tuple containing the logger instance and the experiment directory path.
    """
    results_dir = Path(output_dir)
    exp_dir = increment_path(results_dir / exp_name, exist_ok=False, mkdir=True)
    # Configure logging
    logger.configure_logging(name="benchmarks", level=logging.INFO, log_dir=exp_dir)
    return logger, exp_dir
