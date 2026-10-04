"""Project-wide logging setup."""

import logging

LOGGER_NAME = "riskchallenger"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger within the project-wide logging namespace."""
    logger_name = LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}"
    return logging.getLogger(logger_name)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a simple console handler for local development."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
