"""
Structured logging for Triton compilation and execution.

This module provides utilities for structured logging of Triton operations,
particularly focused on kernel compilation tracing. It includes custom log
formatters, handlers, and helper functions to capture and store detailed
information about Triton kernels, their compilation artifacts, and execution.
"""

import inspect
import atexit
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import triton
from triton.compiler import ASTSource, IRSource


class TritonLogRecord(logging.LogRecord):
    """
    Custom LogRecord class for structured logging of Triton operations.

    Extends the standard LogRecord with additional attributes for storing
    structured metadata and payload information.
    """

    def __init__(self, name, level, pathname, lineno, msg, args, exc_info,
                 metadata=None, payload=None, **kwargs):
        super().__init__(name, level, pathname, lineno, msg, args, exc_info, **kwargs)
        self.metadata: Dict[str, Any] = metadata or {}
        self.payload: Optional[Union[str, Dict[str, Any], list]] = payload


def create_triton_log_record(name=None, level=logging.DEBUG, pathname=None, lineno=None,
                             msg="", args=(), exc_info=None,
                             metadata=None, payload=None, **kwargs):
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

    record = TritonLogRecord(name, level, pathname, lineno, msg, args, exc_info,
                             metadata=metadata, payload=payload, **kwargs)
    return record


TEXT_FILE_EXTENSIONS = [".ttir", ".ttgir", ".llir", ".ptx", ".amdgcn", ".json"]
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB limit for file content extraction

log = logging.getLogger(__name__)


triton_trace_log = logging.getLogger("triton.trace")
TRITON_TRACE_HANDLER = None
triton_trace_folder = os.getenv("TRITON_TRACE", None)


TORCH_INSTALLED = True
try:
    import torch
except ImportError:
    TORCH_INSTALLED = False

def maybe_enable_debug_logging():
    if os.getenv("TRITONPARSE_DEBUG", None) in ["1", "true", "True"]:
        log_handler = logging.StreamHandler()
        log_handler.setLevel(logging.DEBUG)
        log_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        log.setLevel(logging.DEBUG)
        log.addHandler(log_handler)


def _init_logs():
    """
    Initialize the logging system for Triton tracing.

    Sets up the global trace handler and configures the logger with appropriate
    formatter and log level.
    """
    global TRITON_TRACE_HANDLER
    global triton_trace_folder
    if TRITON_TRACE_HANDLER is None:
        TRITON_TRACE_HANDLER = TritonTraceHandler(triton_trace_folder)
        triton_trace_log.setLevel(logging.DEBUG)
        TRITON_TRACE_HANDLER.setFormatter(TritonJsonFormatter())
        triton_trace_log.addHandler(TRITON_TRACE_HANDLER)


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
    frames = []
    try:
        from torch.utils._traceback import CapturedTraceback
    except ImportError:
        return []

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


class TritonJsonFormatter(logging.Formatter):
    """
    Format log records as JSON for Triton compilation tracing.

    This formatter converts log records with metadata and payload into NDJSON format,
    suitable for structured logging and later analysis. It handles special attributes
    added by the Triton tracing system, such as metadata dictionaries and payload data.
    """

    def format(self, record: logging.LogRecord):
        log_entry = record.metadata
        payload = record.payload

        log_entry["timestamp"] = self.formatTime(
            record, "%Y-%m-%dT%H:%M:%S.%fZ")
        if payload is not None:
            log_entry["payload"] = json.loads(payload)
        # NDJSON format requires a newline at the end of each line
        return json.dumps(log_entry, separators=(",", ":")) + "\n"


class TritonTraceHandler(logging.StreamHandler):
    """
    A handler for Triton compilation tracing that outputs NDJSON files.

    This handler creates and manages log files for Triton kernel compilation traces.
    It supports creating new files for different compilation events and handles
    proper cleanup of file resources. When running in a distributed environment,
    it automatically adds rank information to filenames.
    """

    def __init__(self, root_dir: Optional[str], prefix="dedicated_log_triton_trace_"):
        logging.Handler.__init__(self)
        self.root_dir = root_dir
        self.prefix = prefix
        self.stream = None
        self.first_record = True
        # Register cleanup handler for program exit
        atexit.register(self._cleanup)

    def emit(self, record):
        try:
            if self.stream is None:
                if self.root_dir is None:
                    # For meta internal runs, we use the /logs directory by default
                    # reference implementation
                    # https://github.com/pytorch/pytorch/blob/5fe58ab5bd9e14cce3107150a9956a2ed40d2f79/torch/_logging/_internal.py#L1071
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
                        elif not torch._utils_internal.justknobs_check(
                            "pytorch/trace:enable"
                        ):
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

                # Create directory and file
                if self.root_dir is not None:
                    os.makedirs(self.root_dir, exist_ok=True)
                    ranksuffix = ""
                    if TORCH_INSTALLED:
                        import torch.distributed as dist
                        if dist.is_available() and dist.is_initialized():
                            ranksuffix = f"rank_{dist.get_rank()}_"
                    filename = f"{self.prefix}{ranksuffix}"
                    self._ensure_stream_closed()

                    self.stream = open(
                        os.path.join(self.root_dir, f"{filename}.ndjson"),
                        mode="a+",
                    )
                    log.debug("TritonTraceHandler: logging to %s",
                              self.stream.name)
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


def trace_structured_triton(
    name: str,
    metadata_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    *,
    payload_fn: Optional[Callable[[], Optional[Union[str, object]]]] = None,
):
    """
    Record structured trace information for Triton kernel compilation.

    This function is the main entry point for logging structured trace events
    in the Triton system. It handles initialization of the logging system if needed,
    creates new log files, and formats the trace data with metadata
    and payload information.

    Args:
        name (str): Name of the trace event (e.g., "compilation", "execution")
        metadata_fn (Callable): Function that returns a dictionary of metadata to include
                               in the trace record
        payload_fn (Callable): Function that returns the payload data (can be a string,
                              dictionary, or other serializable object)
    """
    global TRITON_TRACE_HANDLER
    global triton_trace_folder
    # Initialize logging if needed
    if not triton_trace_log.handlers or TRITON_TRACE_HANDLER is None:
        # Clear existing handlers
        for handler in list(triton_trace_log.handlers):
            triton_trace_log.removeHandler(handler)

        TRITON_TRACE_HANDLER = None
        _init_logs()

    # Create new file if TRITON_TRACE_HANDLER exists and has a stream
    if TRITON_TRACE_HANDLER and TRITON_TRACE_HANDLER.stream is not None:
        TRITON_TRACE_HANDLER.close()
        TRITON_TRACE_HANDLER.stream = None

    if metadata_fn is None:
        def metadata_fn(): return {}
    if payload_fn is None:
        def payload_fn(): return None

    metadata_dict: Dict[str, Any] = {"event_type": name}
    metadata_dict["pid"] = os.getpid()
    custom_metadata = metadata_fn()
    if custom_metadata:
        metadata_dict.update(custom_metadata)

    metadata_dict["stack"] = get_stack_trace()

    # Log the record using our custom LogRecord
    payload = payload_fn()
    # Use a custom factory to create the record with simplified parameters
    record = create_triton_log_record(metadata=metadata_dict, payload=payload)
    # Log the custom record
    triton_trace_log.handle(record)


def extract_python_source_info(trace_data: Dict[str, Any], source: Any):
    """
    Extract Python source code information from the source object and add it to trace_data.

    This function uses Python's inspect module to extract source code information
    from the provided source object (typically an ASTSource or IRSource instance).
    It adds file path, line numbers, and the actual source code to the trace_data.

    Args:
        trace_data (Dict[str, Any]): Dictionary to store extracted information
        source (Union[ASTSource, IRSource]): Source object containing kernel function information
    """
    if not source:
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
    if not metadata_group:
        return
    for ir_filename, file_path in metadata_group.items():
        # Add file path to trace data
        trace_data["file_path"][ir_filename] = file_path

        # Check if this is a text file we can read
        if any(ir_filename.endswith(ext) for ext in TEXT_FILE_EXTENSIONS):
            try:
                # Check file size before reading to avoid memory issues
                file_size = os.path.getsize(file_path)
                if file_size > MAX_FILE_SIZE:
                    trace_data["file_content"][
                        ir_filename
                    ] = f"<file too large: {file_size} bytes>"
                    continue

                with open(file_path, "r") as f:
                    trace_data["file_content"][ir_filename] = f.read()
            except (IOError, UnicodeDecodeError, OSError) as e:
                # add more specific error type
                trace_data["file_content"][
                    ir_filename] = f"<error reading file: {str(e)}>"
                log.debug(f"Error reading file {file_path}: {e}")


def maybe_trace_triton(
    src: Union[ASTSource, IRSource],
    metadata: Dict[str, Any],
    metadata_group: Dict[str, Any],
    times: Any,
    event_type: str = "compilation",
    cache_hit: bool = False,
):
    """
    Collect and trace Triton kernel compilation information for debugging and profiling.

    This function gathers metadata, IR files, and source code information about a Triton
    kernel compilation, then logs it through the tracing system if tracing is enabled.
    It collects information from multiple sources:
    1. JSON metadata file (if provided)
    2. PyTorch compilation context (if available)
    3. IR and other compilation artifact files
    4. Python source code of the kernel function

    This function is designed to be used as a CompilationListener in triton.knobs.compilation.listener,
    which now accepts a list of listeners.

    Args:
        src (Union[ASTSource, IRSource]): Source object containing kernel information
        metadata (Dict[str, Any]): Dictionary containing metadata for the compilation
        metadata_group (Dict[str, Any]): Dictionary mapping filenames to file paths for all compilation artifacts
        times (CompileTimes): Object containing timing information for the compilation
        event_type (str): Type of event being traced (default: "compilation")
        cache_hit (bool): Whether the compilation was a cache hit (default: False)

    Returns:
        Dict[str, Any]: Dictionary containing all collected trace data, even if tracing is disabled
    """
    # Initialize a dictionary with defaultdict to avoid key errors
    trace_data = defaultdict(dict)
    # Add cache_hit to metadata
    trace_data["metadata"]["cache_hit"] = cache_hit
    if not metadata:
        metadata_path = next(
            (Path(p) for c, p in metadata_group.items() if c.endswith(".json"))
        )
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            # Copy all metadata fields to the trace data
            for key, value in metadata.items():
                trace_data["metadata"][key] = value
    # Handle torch._guards which might not be recognized by type checker
    if TORCH_INSTALLED:
        trace_id = torch._guards.CompileContext.current_trace_id()  # type: ignore
    else:
        trace_id = None
    cid = trace_id.compile_id if trace_id else None
    if cid is not None:
        for attr_name in ["compiled_autograd_id", "frame_id", "frame_compile_id"]:
            attr_value = getattr(cid, attr_name, None)
            if attr_value is not None:
                trace_data["pt_info"][attr_name] = attr_value
    if trace_id:
        trace_data["pt_info"]["attempt"] = trace_id.attempt
    # Extract content from all IR and other files in the metadata group
    extract_file_content(trace_data, metadata_group)

    # Extract Python source code information if available
    extract_python_source_info(trace_data, src)

    # Log the collected information through the tracing system
    trace_structured_triton(
        event_type,
        payload_fn=lambda: json.dumps(trace_data),
    )

    return trace_data


def init(trace_folder: Optional[str] = None):
    """
    Initialize the structured logging system for Triton compilation.

    This function sets up the logging system for Triton kernel compilation traces,
    including the TRITON_TRACE environment variable and the TRITON_TRACE_HANDLER.

    Args:
        trace_folder (Optional[str]): The folder to store the trace files.
    """
    global triton_trace_folder
    maybe_enable_debug_logging()
    if triton_trace_folder is not None and trace_folder is not None:
        log.info("Conflict settings: TRITON_TRACE is already set to %s, we will use provided trace_folder(%s) instead.",
                 triton_trace_folder, trace_folder)
    triton_trace_folder = trace_folder
    triton.knobs.compilation.listener = maybe_trace_triton
