import importlib
import inspect
import logging
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Union

from triton.compiler import ASTSource, IRSource

log = logging.getLogger(__name__)

TEXT_FILE_EXTENSIONS = [".ttir", ".ttgir", ".llir", ".ptx", ".amdgcn", ".json"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit for file content extraction

triton_trace_log = logging.getLogger("tritonparse")
TORCH_INSTALLED = True
if importlib.util.find_spec("torch") is not None:
    TORCH_INSTALLED = True
    from torch.utils._traceback import CapturedTraceback
else:
    TORCH_INSTALLED = False

# enable debug logging for tritonparse itself
TRITONPARSE_DEBUG = os.getenv("TRITONPARSE_DEBUG", None) in ["1", "true", "True"]


class TritonLogRecord(logging.LogRecord):
    """
    Custom LogRecord class for structured logging of Triton operations.

    Extends the standard LogRecord with additional attributes for storing
    structured metadata and payload information.
    """

    def __init__(
        self,
        name,
        level,
        pathname,
        lineno,
        msg,
        args,
        exc_info,
        metadata=None,
        payload=None,
        **kwargs,
    ):
        super().__init__(name, level, pathname, lineno, msg, args, exc_info, **kwargs)
        self.metadata: Dict[str, Any] = metadata or {}
        self.payload: Optional[Union[str, Dict[str, Any], list]] = payload


def create_triton_log_record(
    name=None,
    level=logging.DEBUG,
    pathname=None,
    lineno=None,
    msg="",
    args=(),
    exc_info=None,
    metadata=None,
    payload=None,
    **kwargs,
):
    """
    Factory method to create TritonLogRecord instances with sensible defaults.

    Args:
        name (str, optional): Logger name. Defaults to triton_trace_log.name.
        level (int, optional): Log level. Defaults to DEBUG.
        pathname (str, optional): Path to the file where the log call was made. Defaults to current file.
        lineno (int, optional): Line number where the log call was made. Defaults to current line.
        msg (str, optional): Log message. Defaults to empty string.
        args (tuple, optional): Arguments to interpolate into the message. Defaults to empty tuple.
        exc_info (optional): Exception information. Defaults to None.
        metadata (Dict[str, Any], optional): Structured metadata for the log record. Defaults to empty dict.
        payload (optional): Payload data. Defaults to None.
        **kwargs: Additional keyword arguments for LogRecord

    Returns:
        TritonLogRecord: A custom log record with structured data
    """
    if pathname is None:
        pathname = __file__
    if lineno is None:
        lineno = inspect.currentframe().f_back.f_lineno
    if name is None:
        name = triton_trace_log.name

    record = TritonLogRecord(
        name,
        level,
        pathname,
        lineno,
        msg,
        args,
        exc_info,
        metadata=metadata,
        payload=payload,
        **kwargs,
    )
    return record


def convert(obj):
    """
    Recursively converts dataclasses, dictionaries, and lists to their serializable forms.

    Args:
        obj: The object to convert, which can be a dataclass instance, dictionary, list, or any other type

    Returns:
        A serializable version of the input object where dataclasses are converted to dictionaries
    """
    if is_dataclass(obj):
        return convert(
            asdict(obj)
        )  # Convert dataclass to dict and then process that dict
    elif isinstance(obj, dict):
        return {
            k: convert(v) for k, v in obj.items()
        }  # Process each key-value pair recursively
    elif isinstance(obj, list):
        return [convert(i) for i in obj]  # Process each list item recursively
    else:
        return obj  # Return primitive types as-is


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


def extract_python_source_info(
    trace_data: Dict[str, Any], source: Union[ASTSource, IRSource]
):
    """
    Extract Python source code information from the source object and add it to trace_data.

    This function uses Python's inspect module to extract source code information
    from the provided source object (typically an ASTSource or IRSource instance).
    It adds file path, line numbers, and the actual source code to the trace_data.

    Args:
        trace_data (Dict[str, Any]): Dictionary to store extracted information
        source (Union[ASTSource, IRSource]): Source object containing kernel function information
    """
    # @TODO: add support for IRSource
    if isinstance(source, IRSource):
        return
    # Get the original Python source code for the kernel
    target_fn = source.fn.fn
    python_source_file = inspect.getfile(target_fn)
    source_lines, start_line_number = inspect.getsourcelines(target_fn)
    end_line_number = start_line_number + len(source_lines)

    trace_data["python_source"] = {
        "file_path": python_source_file,
        "start_line": start_line_number,
        "end_line": end_line_number,
        "code": inspect.getsource(target_fn),
    }


def extract_file_content(trace_data: Dict[str, Any], metadata_group: Dict[str, str]):
    """
    Extract file content from metadata_group and add it to trace_data.

    Args:
        trace_data (Dict): Dictionary to store extracted information
        metadata_group (Dict): Dictionary mapping filenames to file paths
    """
    for ir_filename, file_path in metadata_group.items():
        # Add file path to trace data
        trace_data["file_path"][ir_filename] = file_path

        # Check if this is a text file we can read
        if any(ir_filename.endswith(ext) for ext in TEXT_FILE_EXTENSIONS):
            try:
                # Check file size before reading to avoid memory issues
                file_size = os.path.getsize(file_path)
                if file_size > MAX_FILE_SIZE:
                    trace_data["file_content"][ir_filename] = (
                        f"<file too large: {file_size} bytes>"
                    )
                    continue

                with open(file_path, "r") as f:
                    trace_data["file_content"][ir_filename] = f.read()
            except (UnicodeDecodeError, OSError) as e:
                # add more specific error type
                trace_data["file_content"][ir_filename] = (
                    f"<error reading file: {str(e)}>"
                )
                log.debug(f"Error reading file {file_path}: {e}")
