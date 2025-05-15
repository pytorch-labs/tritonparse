import importlib
import logging
import os

log = logging.getLogger(__name__)


TORCH_INSTALLED = True
if importlib.util.find_spec("torch") is not None:
    TORCH_INSTALLED = True
else:
    TORCH_INSTALLED = False

# enable debug logging for tritonparse itself
TRITONPARSE_DEBUG = os.getenv("TRITONPARSE_DEBUG", None) in ["1", "true", "True"]


def maybe_enable_debug_logging():
    """
    This logging is for logging module itself, not for logging the triton compilation.
    """
    if TRITONPARSE_DEBUG and not log.hasHandlers():
        log_handler = logging.StreamHandler()
        log_handler.setLevel(logging.DEBUG)
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        log.setLevel(logging.DEBUG)
        log.addHandler(log_handler)
