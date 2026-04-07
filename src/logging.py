import logging
import sys

from src.config import settings


def setup_logging() -> None:
    logging.getLogger().handlers.clear()

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    # Suppress noisy logs
    for logger_name in ["google_genai", "httpx", "httpcore"]:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
