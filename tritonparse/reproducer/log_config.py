import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the reproducer package."""
    logger = logging.getLogger("tritonparse.reproducer")
    if logger.hasHandlers():
        # Avoid adding handlers multiple times if the function is called again.
        return

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s - %(levelname)s - %(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
