#  Copyright (c) Meta Platforms, Inc. and affiliates.

import atexit
import gzip
import hashlib
import importlib
import inspect
import json
import logging
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

import triton

log = logging.getLogger(__name__)

TEXT_FILE_EXTENSIONS = [".ttir", ".ttgir", ".llir", ".ptx", ".amdgcn", ".json"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit for file content extraction
# Enable ndjson output. json format is only for debugging purpose.
TRITONPARSE_NDJSON = os.getenv("TRITONPARSE_NDJSON", "1") in ["1", "true", "True"]
triton_trace_log = logging.getLogger("tritonparse_trace")
# The folder to store the triton trace log.
triton_trace_folder = os.environ.get("TRITON_TRACE", None)
# Enable debug logging for tritonparse itself
TRITONPARSE_DEBUG = os.getenv("TRITONPARSE_DEBUG", None) in ["1", "true", "True"]
# The compilation information will be stored to /logs/DEFAULT_TRACE_FILE_PREFIX by default
# unless other flags disable or set another store. Add USER to avoid permission issues in shared servers.
DEFAULT_TRACE_FILE_PREFIX = (
    f"dedicated_log_triton_trace_{os.getenv('USER', 'unknown')}_"
)
# Enable launch trace. WARNNING: it will overwrite launch_metadata for each triton kernel.
TRITON_TRACE_LAUNCH = os.getenv("TRITON_TRACE_LAUNCH", None) in ["1", "true", "True"]
# Enable tensor blob storage
TRITONPARSE_SAVE_TENSOR_BLOBS = os.getenv("TRITONPARSE_SAVE_TENSOR_BLOBS", "0") in ["1", "true", "True"]
# Tensor size limit in bytes (default 10GB)
TRITONPARSE_TENSOR_SIZE_LIMIT = int(os.getenv("TRITONPARSE_TENSOR_SIZE_LIMIT", str(10 * 1024 * 1024 * 1024)))

TRITON_TRACE_HANDLER = None
# Global tensor blob manager instance
TENSOR_BLOB_MANAGER = None

if importlib.util.find_spec("torch") is not None:
    TORCH_INSTALLED = True
    import torch
    from torch.utils._traceback import CapturedTraceback
else:
    TORCH_INSTALLED = False


class TensorBlobManager:
    """
    Manager for storing tensor data as content-addressed blobs.
    
    Uses BLAKE2b hashing for content addressing and stores blobs in a two-level
    directory structure to avoid filesystem limitations with large numbers of files.
    """
    
    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = None
        self.hash_to_path_cache = {}  # In-memory cache for hash -> path mapping
        if root_dir:
            self.set_root_dir(root_dir)
    
    def set_root_dir(self, root_dir: str):
        """Set the root directory for blob storage."""
        self.root_dir = Path(root_dir) / "saved_tensors"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        log.debug(f"TensorBlobManager: using root directory {self.root_dir}")
    
    def _compute_hash(self, data: bytes) -> str:
        """Compute BLAKE2b hash of the data."""
        return hashlib.blake2b(data).hexdigest()
    
    def _get_blob_path(self, hash_hex: str) -> Path:
        """Get the file path for a given hash using two-level directory structure."""
        if not self.root_dir:
            raise ValueError("Root directory not set")
        
        # Two-level directory: first 2 chars / full_hash.bin
        subdir = hash_hex[:2]
        filename = f"{hash_hex}.bin"
        return (self.root_dir / subdir / filename).resolve()
    
    def _get_tensor_size_bytes(self, tensor) -> int:
        """Get tensor size in bytes before serialization."""
        if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
            return tensor.numel() * tensor.element_size()
        return 0
    
    def save_tensor_blob(self, tensor) -> Dict[str, Any]:
        """
        Save tensor as a blob and return metadata.
        
        Args:
            tensor: PyTorch tensor to save
            
        Returns:
            Dictionary with blob metadata or error information:
            - Success: {'tensor_hash': str, 'blob_path': str, 'blob_size': int, 'serialization_method': str}
            - Error: {'error': str, 'tensor_hash': None}
        """
        if not self.root_dir:
            return {'error': 'Blob storage not initialized', 'tensor_hash': None}
        
        try:
            # Check tensor size before serialization
            tensor_size = self._get_tensor_size_bytes(tensor)
            if tensor_size > TRITONPARSE_TENSOR_SIZE_LIMIT:
                log.warning(
                    f"Tensor size {tensor_size} bytes exceeds limit {TRITONPARSE_TENSOR_SIZE_LIMIT} bytes, skipping blob storage"
                )
                return {
                    'error': f'Tensor size {tensor_size} bytes exceeds limit {TRITONPARSE_TENSOR_SIZE_LIMIT} bytes',
                    'tensor_hash': None
                }
            
            # Serialize tensor using torch.save
            # TODO: Consider async serialization for very large tensors to avoid blocking
            import io
            buffer = io.BytesIO()
            if TORCH_INSTALLED:
                torch.save(tensor.cpu(), buffer)
            else:
                return {'error': 'PyTorch not available for tensor serialization', 'tensor_hash': None}
            
            blob_data = buffer.getvalue()
            hash_hex = self._compute_hash(blob_data)
            
            # Check if we already have this blob
            if hash_hex in self.hash_to_path_cache:
                blob_path = self.hash_to_path_cache[hash_hex]
                if blob_path.exists():
                    return {
                        'tensor_hash': hash_hex,
                        'blob_path': str(blob_path),
                        'blob_size': len(blob_data),
                        'serialization_method': 'torch_save'
                    }
            
            # Create blob file
            blob_path = self._get_blob_path(hash_hex)
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic write using temporary file + rename
            with tempfile.NamedTemporaryFile(
                mode='wb', 
                dir=blob_path.parent, 
                prefix=f".tmp_{hash_hex}_",
                delete=False
            ) as tmp_file:
                tmp_file.write(blob_data)
                tmp_path = Path(tmp_file.name)
            
            # Atomic rename
            tmp_path.rename(blob_path)
            
            # Cache the path
            self.hash_to_path_cache[hash_hex] = blob_path
            
            log.debug(f"Saved tensor blob: {hash_hex} -> {blob_path}")
            
            return {
                'tensor_hash': hash_hex,
                'blob_path': str(blob_path),
                'blob_size': len(blob_data),
                'serialization_method': 'torch_save'
            }
            
        except Exception as e:
            error_msg = f"Failed to save tensor blob: {str(e)}"
            log.error(error_msg)
            return {'error': error_msg, 'tensor_hash': None}


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
    # 1. primitives that JSON already supports  -------------------------------
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj

    if isinstance(obj, float):
        # JSON spec forbids NaN/Infinity – keep precision but stay valid
        if math.isfinite(obj):
            return obj
        return str(obj)  # "NaN", "inf", "-inf"

    # 2. simple containers ----------------------------------------------------
    if isinstance(obj, (list, tuple)):
        return [convert(x) for x in obj]

    if isinstance(obj, (set, frozenset)):
        return [convert(x) for x in sorted(obj, key=str)]

    if isinstance(obj, Mapping):
        return {str(k): convert(v) for k, v in obj.items()}

    # 3. time, enum, path, bytes ---------------------------------------------
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, Enum):
        return convert(obj.value)

    if isinstance(obj, Path):
        return str(obj)

    if is_dataclass(obj):
        return convert(
            asdict(obj)
        )  # Convert dataclass to dict and then process that dict
    log.warning(f"Unknown type: {type(obj)}")
    return str(obj)  # Return primitive types as-is


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


def extract_python_source_info(trace_data: Dict[str, Any], source):
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
    from triton.compiler import IRSource

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


def extrac_metadata_from_src(trace_data, src):
    from triton._C.libtriton import get_cache_invalidating_env_vars

    env_vars = get_cache_invalidating_env_vars()
    # extra_options = src.parse_options()
    # options = backend.parse_options(dict(options or dict(), **extra_options))

    # trace_data["extra_options"] = extra_options
    trace_data["metadata"].update(
        {
            "env": env_vars,
            "src_attrs": src.attrs if hasattr(src, "attrs") else {},
            "src_cache_key": src.fn.cache_key if hasattr(src, "fn") else "",
            "src_constants": src.constants if hasattr(src, "constants") else {},
        }
    )


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
            return json.dumps(clean_log_entry, indent=2)
        else:
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
        self, root_dir: Optional[str] = None, prefix=DEFAULT_TRACE_FILE_PREFIX
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
        if should_set_root_dir:
            if not os.path.exists(TRACE_LOG_DIR):
                log.info(
                    "TritonTraceHandler: disabled because %s does not exist",
                    TRACE_LOG_DIR,
                )
            elif not os.access(TRACE_LOG_DIR, os.W_OK):
                log.info(
                    "TritonTraceHandler: disabled because %s is not writeable",
                    TRACE_LOG_DIR,
                )
            else:
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


def init_logs():
    """
    Initialise tritonparse's logging system.

    Requirements handled:
    1. First call may or may not pass `trace_folder`.
    2. A later call *can* pass `trace_folder` and must activate an
       existing handler whose `root_dir` was None.
    3. When tracing is disabled (no writable directory), prevent the
       empty                                     →
           DEBUG:tritonparse_trace:
       lines by blocking propagation to the root logger.
    """
    global TRITON_TRACE_HANDLER, triton_trace_folder, TENSOR_BLOB_MANAGER

    # Basic logger settings (safe to run on every call)
    triton_trace_log.setLevel(logging.DEBUG)
    triton_trace_log.propagate = False  # stops bubbling to root logger. see 3)
    # 1) Create the handler on first use (root_dir may be None)
    if TRITON_TRACE_HANDLER is None:
        TRITON_TRACE_HANDLER = TritonTraceHandler(triton_trace_folder)
    # 2) If the handler has no root_dir but we now have
    #    `triton_trace_folder`, fill it in.
    if TRITON_TRACE_HANDLER.root_dir is None and triton_trace_folder is not None:
        TRITON_TRACE_HANDLER.root_dir = triton_trace_folder
    # 3) Re-evaluate whether we have a writable directory
    #    (`get_root_dir()` also checks /logs logic, permissions, etc.)
    root_dir = TRITON_TRACE_HANDLER.get_root_dir()
    if root_dir is None:
        # Tracing still disabled: ensure the handler is NOT attached
        if TRITON_TRACE_HANDLER in triton_trace_log.handlers:
            triton_trace_log.removeHandler(TRITON_TRACE_HANDLER)
        return  # quiet exit, no blank lines
    # 4) Tracing is enabled: attach the handler (if not already
    #    attached) and set the JSON formatter.
    if TRITON_TRACE_HANDLER not in triton_trace_log.handlers:
        TRITON_TRACE_HANDLER.setFormatter(TritonJsonFormatter())
        triton_trace_log.addHandler(TRITON_TRACE_HANDLER)
    
    # Initialize tensor blob manager if enabled
    if TRITONPARSE_SAVE_TENSOR_BLOBS:
        if TENSOR_BLOB_MANAGER is None:
            TENSOR_BLOB_MANAGER = TensorBlobManager()
        
        # Set or update root directory for blob storage
        if root_dir and TENSOR_BLOB_MANAGER.root_dir is None:
            TENSOR_BLOB_MANAGER.set_root_dir(root_dir)


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

    if metadata_fn is None:

        def metadata_fn():
            return {}

    if payload_fn is None:

        def payload_fn():
            return None

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


def maybe_trace_triton(
    src,
    metadata: Dict[str, Any],
    metadata_group: Dict[str, str],
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
            trace_data["metadata"].update(metadata)
    else:
        trace_data["metadata"].update(metadata)
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
    extrac_metadata_from_src(trace_data, src)

    # Add timing information if available
    if times:
        trace_data["times"] = times
    # Log the collected information through the tracing system
    trace_structured_triton(
        event_type,
        payload_fn=lambda: json.dumps(convert(trace_data)),
    )

    return trace_data


from triton.knobs import LaunchHook, JITHook


def extract_arg_info(arg_dict):
    """
    Extract detailed information from kernel arguments, especially for PyTorch tensors.
    
    Args:
        arg_dict: Dictionary of kernel arguments
        
    Returns:
        Dictionary with extracted argument information including tensor properties
    """
    global TENSOR_BLOB_MANAGER
    
    extracted_args = {}
    
    for arg_name, arg_value in arg_dict.items():
        arg_info = {}
        
        # Check if it's a PyTorch tensor
        if hasattr(arg_value, 'shape') and hasattr(arg_value, 'dtype'):
            arg_info['type'] = 'tensor'
            arg_info['shape'] = list(arg_value.shape)
            arg_info['dtype'] = str(arg_value.dtype)
            arg_info['device'] = str(arg_value.device)
            arg_info['stride'] = list(arg_value.stride())
            arg_info['numel'] = arg_value.numel()
            arg_info['is_contiguous'] = arg_value.is_contiguous()
            arg_info['element_size'] = arg_value.element_size()
            arg_info['storage_offset'] = arg_value.storage_offset()
            # Memory usage in bytes
            arg_info['memory_usage'] = arg_value.numel() * arg_value.element_size()
            # Add data_ptr for memory tracking (optional)
            if hasattr(arg_value, 'data_ptr'):
                arg_info['data_ptr'] = hex(arg_value.data_ptr())
            
            # Add tensor blob storage if enabled
            if TRITONPARSE_SAVE_TENSOR_BLOBS and TENSOR_BLOB_MANAGER is not None:
                blob_info = TENSOR_BLOB_MANAGER.save_tensor_blob(arg_value)
                arg_info.update(blob_info)
                
        # Handle scalar values
        elif isinstance(arg_value, (int, float, bool)):
            arg_info['type'] = type(arg_value).__name__
            arg_info['value'] = arg_value
        # Handle strings
        elif isinstance(arg_value, str):
            arg_info['type'] = 'str'
            arg_info['value'] = arg_value
            arg_info['length'] = len(arg_value)
        # Handle other types
        else:
            arg_info['type'] = type(arg_value).__name__
            # Try to convert to string for logging, but be safe about it
            try:
                arg_info['repr'] = str(arg_value)
                if len(arg_info['repr']) > 200:  # Truncate very long representations
                    arg_info['repr'] = arg_info['repr'][:200] + "..."
            except:
                arg_info['repr'] = f"<{type(arg_value).__name__} object>"
                
        extracted_args[arg_name] = arg_info
    
    return extracted_args


def add_launch_metadata(grid, metadata, arg_dict):
    # Extract detailed argument information
    extracted_args = extract_arg_info(arg_dict)
    return {"launch_metadata_tritonparse": (grid, metadata, extracted_args)}


class JITHookImpl(JITHook):
    """
    JIT Hook implementation that overrides or sets the launch_metadata function for Triton kernels.
    
    This hook is essential for capturing detailed kernel launch information beyond the basic
    metadata (like kernel name) that Triton provides by default. Without setting a custom
    launch_metadata function, only minimal launch information is available as shown in:
    https://github.com/triton-lang/triton/blob/7ce287dc24b43476cdeb30529089ac361564505d/python/triton/compiler/compiler.py#L504
    
    By intercepting the JIT compilation process and setting a custom launch_metadata function,
    we can capture comprehensive runtime information including grid parameters, kernel metadata,
    and argument dictionaries for detailed analysis and logging.
    """

    def __call__(
        self,
        *,
        key: str,
        repr: str,
        fn,
        compile,
        is_manual_warmup: bool,
        already_compiled: bool,
    ) -> Optional[bool]:
        """
        Override or set the launch_metadata function for the JIT-compiled kernel.
        
        This method is called during the JIT compilation process and allows us to
        inject our custom launch_metadata function that will be used to collect
        detailed kernel launch information.
        
        Args:
            key: Unique identifier for the kernel
            repr: String representation of the kernel
            fn: The JIT function object
            compile: Compilation function
            is_manual_warmup: Whether this is a manual warmup call
            already_compiled: Whether the kernel is already compiled
            
        Returns:
            True to continue with compilation, None/False to skip
        """
        launch_metadata_fn = fn.jit_function.launch_metadata
        if launch_metadata_fn is not None:
            log.warning(
                f"fn {fn} launch_metadata_fn is not None: {launch_metadata_fn}. It will be overridden by tritonparse."
            )
        fn.jit_function.launch_metadata = add_launch_metadata
        return True


class LaunchHookImpl(LaunchHook):
    """
    Launch Hook implementation for capturing and logging kernel launch metadata.
    
    This hook is responsible for intercepting kernel launches and extracting the detailed
    metadata that was set up by the JITHookImpl. It provides entry point for
    kernel execution, allowing comprehensive logging and analysis of kernel launches
    including timing, parameters, and execution context.
    
    The metadata captured includes:
    - Kernel name and function details
    - Grid dimensions and launch parameters  
    - Kernel arguments and their values
    - Stream information
    - Custom metadata added by the launch_metadata function
    """

    def enter(self, metadata):
        """
        Handle kernel launch entry point.
        
        This method is called when a kernel is about to be launched, providing
        access to all the launch metadata for logging, profiling, or analysis.
        metadata format:

                Args:
            metadata: LazyDict containing comprehensive launch information including
                     kernel name, function, stream, grid parameters, and custom data
                     format: {'name': 'add_kernel', 'function': None, 'stream': 0,
                              'launch_metadata_tritonparse': (grid, self.metadata, extracted_args)}
                     where extracted_args contains detailed info for each argument:
                     - For tensors: shape, dtype, device, stride, memory_usage, etc.
                     - For scalars: type and value
                     - For other types: type and string representation
                 defined here:
                 https://github.com/triton-lang/triton/blob/7ce287dc24b43476cdeb30529089ac361564505d/
                 python/triton/compiler/compiler.py#L512.
        """
        trace_data = defaultdict(dict)
        metadata_dict = metadata.get()
        trace_data["name"] = metadata_dict["name"]
        trace_data["function"] = metadata_dict["function"]
        trace_data["stream"] = metadata_dict["stream"]
        launch_metadata_tritonparse = metadata_dict.get("launch_metadata_tritonparse", None)
        if launch_metadata_tritonparse is not None:
            trace_data["grid"] = launch_metadata_tritonparse[0]
            trace_data["metadata"] = launch_metadata_tritonparse[1]
            trace_data["extracted_args"] = launch_metadata_tritonparse[2]  # Now contains detailed arg info
        trace_structured_triton("launch", metadata_fn=lambda: convert(trace_data))
        


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
        log.info(
            "Conflict settings: TRITON_TRACE is already set to %s, we will use provided trace_folder(%s) instead.",
            triton_trace_folder,
            trace_folder,
        )
    if trace_folder is not None:
        triton_trace_folder = trace_folder
    init_logs()
    triton.knobs.compilation.listener = maybe_trace_triton
