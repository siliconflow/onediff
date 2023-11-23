import logging
import os

LOGGING_NAME = "OneDiff"


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[34m",  # Blue
        "INFO": "\033[92m",  # green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[91m",  # Red
    }

    def format(self, record):
        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, "\033[0m")  # Default to Reset color
        return f"{color}{log_message}\033[0m"


def configure_logging(name, level=logging.INFO, log_dir="."):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a console formatter and add it to a console handler
    console_formatter = ColorFormatter(
        fmt="%(levelname)s [%(asctime)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create a file formatter and add it to a file handler if log_dir is provided
    if log_dir and level == logging.DEBUG:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name}.log")
        file_formatter = logging.Formatter(
            fmt="%(levelname)s [%(asctime)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)


def set_logging(name="OneDiff", debug_mode=False, log_dir="."):
    global LOGGING_NAME
    LOGGING_NAME = name

    level = logging.DEBUG if debug_mode else logging.INFO
    configure_logging(name, level, log_dir)
    return logging.getLogger(name)


LOGGER = logging.getLogger(LOGGING_NAME)

if __name__ == "__main__":
    log_directory = "~/workspace/run"
    # # Debug
    # logger_debug = set_logging("test_debug", debug_mode=True, log_dir=log_directory)
    # logger_debug.debug("debug message in debug mode")
    # logger_debug.info("info message in debug mode")

    # Release
    logger_release = set_logging(
        "test_release", debug_mode=False, log_dir=log_directory
    )
    logger_release.debug("debug message in release mode")  # Won't be printed
    logger_release.info("info message in release mode")
    logger_release.warning("warning message in release mode")
    logger_release.error("error message in release mode")
