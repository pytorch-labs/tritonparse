import importlib
import logging
import os

log = logging.getLogger(__name__)


TORCH_INSTALLED = True
if importlib.util.find_spec("torch") is not None:
    TORCH_INSTALLED = True
    from torch.utils._traceback import CapturedTraceback
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


def get_stack_trace(skip=1):
    """
    Get call stack trace for the current execution context.

    Extracts stack trace information using torch's CapturedTraceback utility,
    providing detailed information about each frame in the call stack.

    Args:
        skip (int): Number of frames to skip from the start of the stack

    Returns:
        List[Dict]: List of frame information dictionaries containing line numbers,
                   function names, filenames, and code snippets
    """
    if not TORCH_INSTALLED:
        return []
    frames = []
    for frame in CapturedTraceback.extract(skip=skip).summary():
        frames.append(
            {
                "line": frame.lineno,
                "name": frame.name,
                "filename": frame.filename,
                "loc": frame.line,
            }
        )
    return frames
