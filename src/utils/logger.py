import logging
from src.utils.helper import load_yaml


class CustomFormatter(logging.Formatter):
    """Custom formatter with color support for console logs."""
    COLORS = {
        "DEBUG": "\033[92m",  # Green
        "INFO": "\033[94m",  # Blue
        "WARNING": "\033[38;5;208m",  # Orange
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[1;91m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        log_color = self.COLORS.get(record.levelname, self.RESET)
        log_message = super().format(record)
        return f"{log_color}{log_message}{self.RESET}"


def _level_from_str(level_str: str) -> int:
    """Convert string log level to logging constant."""
    return getattr(logging, level_str.upper(), logging.DEBUG)


def setup_logger(
        name: str = "llm-assistant-for-code-repos",
        level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Configure and return a logger with colored console output and optional file logging.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent double logging

    if not logger.hasHandlers():
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = CustomFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def set_logging_level(logger: logging.Logger, level: int) -> None:
    """Set the logging level for a given logger and its handlers."""
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


# ---------------- Global default logger ----------------
cfg = load_yaml("configs/logger_config.yaml")  # TODO
logger_cfg = cfg.get("logger", {})
logger = setup_logger(
    name=logger_cfg["name"],
    level=_level_from_str(logger_cfg["level"]))

