import logging
import os


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


class ConfigurableLogger:
    def __init__(self, name=None, log_dir=".", debug_mode=False):
        self.name = name or __name__
        self.log_dir = log_dir
        self.debug_mode = debug_mode
        self.configure_logging()
        self.logger = logging.getLogger(self.name)

    def configure_logging(self):
        name = self.name
        debug = self.debug_mode
        log_dir = self.log_dir
        level = logging.DEBUG if debug else logging.INFO
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
                fmt="%(levelname)s [%(asctime)s] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)


LOGGER = logging.getLogger("ondiff")
