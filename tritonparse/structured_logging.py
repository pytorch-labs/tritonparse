import atexit
import importlib
import inspect
import json
import logging
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Union

from triton.compiler import ASTSource, IRSource

log = logging.getLogger(__name__)

TEXT_FILE_EXTENSIONS = [".ttir", ".ttgir", ".llir", ".ptx", ".amdgcn", ".json"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit for file content extraction
# Enable ndjson output. json format is only for debugging purpose.
TRITONPARSE_NDJSON = os.getenv("TRITONPARSE_NDJSON", "1") in ["1", "true", "True"]
triton_trace_log = logging.getLogger("tritonparse")
# enable debug logging for tritonparse itself
TRITONPARSE_DEBUG = os.getenv("TRITONPARSE_DEBUG", None) in ["1", "true", "True"]
TORCH_INSTALLED = True
if importlib.util.find_spec("torch") is not None:
    TORCH_INSTALLED = True
    import torch
    from torch.utils._traceback import CapturedTraceback
else:
    TORCH_INSTALLED = False


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


class TritonJsonFormatter(logging.Formatter):
    """
    Format log records as JSON for Triton compilation tracing.

    This formatter converts log records with metadata and payload into NDJSON format,
    suitable for structured logging and later analysis. It handles special attributes
    added by the tritonparse, such as metadata dictionaries and payload data.
    """

    def format(self, record: logging.LogRecord):
        log_entry = record.metadata
        payload = record.payload

        log_entry["timestamp"] = self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ")
        if payload is not None:
            log_entry["payload"] = json.loads(payload)
        clean_log_entry = convert(log_entry)
        if not TRITONPARSE_NDJSON:
            log.info("TritonJsonFormatter: using JSON format")
            return json.dumps(clean_log_entry, indent=2)
        else:
            log.info("TritonJsonFormatter: using NDJSON format")
            # NDJSON format requires a newline at the end of each line
            return json.dumps(clean_log_entry, separators=(",", ":")) + "\n"


class TritonTraceHandler(logging.StreamHandler):
    """
    A handler for Triton compilation tracing that outputs NDJSON files.

    This handler creates and manages log files for Triton kernel compilation traces.
    It supports creating new files for different compilation events and handles
    proper cleanup of file resources. When running in a distributed environment,
    it automatically adds rank information to filenames.
    """

    def __init__(
        self, root_dir: Optional[str] = None, prefix="dedicated_log_triton_trace_"
    ):
        logging.Handler.__init__(self)
        self.root_dir = root_dir
        self.prefix = prefix
        self.stream = None
        self.first_record = True
        # If the program is unexpected terminated, atexit can ensure  file resources are properly closed and released.
        # it is because we use `self.stream` to keep the opened file stream, if the program is interrupted by some errors, the stream may not be closed.
        atexit.register(self._cleanup)

    def get_root_dir(self):
        # For meta internal runs, we use the /logs directory by default
        # reference implementation
        # https://github.com/pytorch/pytorch/blob/5fe58ab5bd9e14cce3107150a9956a2ed40d2f79/torch/_logging/_internal.py#L1071
        if self.root_dir:
            return self.root_dir
        TRACE_LOG_DIR = "/logs"
        should_set_root_dir = True
        if TORCH_INSTALLED:
            import torch.version as torch_version

            if (
                hasattr(torch_version, "git_version")
                and os.getenv("MAST_HPC_JOB_NAME") is None
            ):
                log.info(
                    "TritonTraceHandler: disabled because not fbcode or conda on mast"
                )
                should_set_root_dir = False
            # TODO: change to tritonparse knob
            elif not torch._utils_internal.justknobs_check("pytorch/trace:enable"):
                log.info(
                    "TritonTraceHandler: disabled because justknobs_check('pytorch/trace:enable') returned False"
                )
                should_set_root_dir = False
        else:
            if not os.path.exists(TRACE_LOG_DIR):
                log.info(
                    "TritonTraceHandler: disabled because %s does not exist",
                    TRACE_LOG_DIR,
                )
                should_set_root_dir = False
            elif not os.access(TRACE_LOG_DIR, os.W_OK):
                log.info(
                    "TritonTraceHandler: disabled because %s is not writeable",
                    TRACE_LOG_DIR,
                )
                should_set_root_dir = False
        if should_set_root_dir:
            self.root_dir = TRACE_LOG_DIR
        return self.root_dir

    def emit(self, record):
        # reference implementation
        # https://github.com/pytorch/pytorch/blob/5fe58ab5bd9e14cce3107150a9956a2ed40d2f79/torch/_logging/_internal.py#L1071
        try:
            if self.stream is None:
                root_dir = self.get_root_dir()
                if root_dir is not None:
                    os.makedirs(root_dir, exist_ok=True)
                    ranksuffix = ""
                    if TORCH_INSTALLED:
                        import torch.distributed as dist

                        if dist.is_available() and dist.is_initialized():
                            ranksuffix = f"rank_{dist.get_rank()}_"
                    filename = f"{self.prefix}{ranksuffix}"
                    self._ensure_stream_closed()
                    log_file_name = os.path.abspath(
                        os.path.join(root_dir, f"{filename}.ndjson")
                    )
                    self.stream = open(
                        log_file_name,
                        mode="a+",
                    )
                    log.debug("TritonTraceHandler: logging to %s", log_file_name)
                else:
                    triton_trace_log.removeHandler(self)
                    return

            if self.stream:
                formatted = self.format(record)
                self.stream.write(formatted)
                self.flush()
        except Exception as e:
            # record exception and ensure resources are released
            log.error(f"Error in TritonTraceHandler.emit: {e}")
            self._ensure_stream_closed()
            self.handleError(record)  # call Handler's standard error handling

    def close(self):
        """Close the current file."""
        self.acquire()
        try:
            try:
                if self.stream:
                    try:
                        self.flush()
                    finally:
                        self.stream.close()
                        self.stream = None
            finally:
                # Solution adopted from PyTorch PR #120289
                logging.StreamHandler.close(self)
        finally:
            self.release()

    def _cleanup(self):
        """Ensure proper cleanup on program exit"""
        if self.stream is not None:
            self.close()

    def _ensure_stream_closed(self):
        """ensure stream is closed"""
        if self.stream is not None:
            try:
                self.flush()
            finally:
                self.stream.close()
                self.stream = None
