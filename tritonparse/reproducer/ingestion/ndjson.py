import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _iter_events(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON line: %s", line)
                continue


def _index_compilations(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for e in events:
        if e.get("event_type") != "compilation":
            continue
        payload = e.get("payload") or {}
        meta = payload.get("metadata") or {}
        h = meta.get("hash")
        if h:
            idx[h] = e
    logger.debug("Indexed %d compilation events.", len(idx))
    return idx


def _get_launches(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    launches = [e for e in events if e.get("event_type") == "launch"]
    logger.debug("Found %d launch events in total.", len(launches))
    return launches


def _resolve_kernel_info(
    launch: Dict[str, Any], comp_idx: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Resolve kernel info (path, name, code) from a launch event."""
    # In new format, launch has top-level compilation_metadata, not payload.*
    comp_meta = (
        launch.get("compilation_metadata")
        or launch.get("payload", {}).get("compilation_metadata")
        or {}
    )
    h = comp_meta.get("hash")
    if not h:
        logger.warning("Could not find compilation hash in launch event.")
        return {}
    comp = comp_idx.get(h, {})
    if not comp:
        logger.warning("Could not resolve compilation hash '%s' to a compilation event.", h)
        return {}

    payload = comp.get("payload") or {}
    py_source = payload.get("python_source") or {}
    code = py_source.get("code", "")

    # Extract file path and function name
    file_path = py_source.get("file_path")
    # The function name is in the compilation metadata payload
    func_name = (comp.get("payload", {}).get("metadata") or {}).get("name")

    # find '@triton.jit' and slice the string
    jit_marker = "@triton.jit"
    jit_pos = code.find(jit_marker)
    if jit_pos != -1:
        code = code[jit_pos:]
        logger.debug("Extracted kernel source starting from '@triton.jit'.")

    info = {
        "file_path": file_path,
        "function_name": func_name,
        "source_code": code,
    }
    logger.debug("Resolved kernel info: %s", info)
    return info


def _pack_args(args: Dict[str, Any]) -> Dict[str, Any]:
    packed = {}
    for k, v in args.items():
        t = v.get("type") if isinstance(v, dict) else None
        if t == "tensor":
            packed[k] = {
                "type": "tensor",
                "shape": v.get("shape") if isinstance(v, dict) else None,
                "dtype": v.get("dtype") if isinstance(v, dict) else None,
                "device": v.get("device") if isinstance(v, dict) else None,
                "stride": v.get("stride") if isinstance(v, dict) else None,
                "is_contiguous": (
                    v.get("is_contiguous") if isinstance(v, dict) else None
                ),
                "numel": v.get("numel") if isinstance(v, dict) else None,
            }
        else:
            # scalar / NoneType etc
            if isinstance(v, dict):
                packed[k] = {
                    "type": v.get("type"),
                    "value": v.get("value", v.get("repr")),
                }
            else:
                packed[k] = {
                    "type": None,
                    "value": v,
                }
    return packed


# Sentinel and helper to normalize extracted argument values
_SKIP = object()


def _decode_arg(raw: Any):
    if not isinstance(raw, dict):
        return raw
    t = raw.get("type")
    if t == "tensor":
        return _SKIP
    if t == "NoneType":
        return None
    return raw.get("value", raw.get("repr"))


def build_context_bundle(ndjson_path: str, launch_index: int = 0) -> Dict[str, Any]:
    logger.debug("Reading events from '%s'...", ndjson_path)
    events = list(_iter_events(ndjson_path))
    launches = _get_launches(events)
    if not launches:
        raise RuntimeError("No launch events found in NDJSON.")
    if launch_index < 0 or launch_index >= len(launches):
        logger.error(
            "launch_index out of range: %d (total %d)", launch_index, len(launches)
        )
        raise IndexError(
            f"launch_index out of range: {launch_index} (total {len(launches)})"
        )
    logger.info("Targeting launch event at index: %d", launch_index)
    launch = launches[launch_index]
    comp_idx = _index_compilations(events)
    kernel_info = _resolve_kernel_info(launch, comp_idx)
    if not kernel_info.get("file_path") or not kernel_info.get("function_name"):
        raise RuntimeError(
            "Could not resolve kernel file path or function name from NDJSON."
            " The import-based strategy cannot proceed."
        )

    # flatten launch fields (support both formats)
    grid = launch.get("grid") or (launch.get("payload", {})).get("grid")
    comp_meta = (
        launch.get("compilation_metadata")
        or (launch.get("payload", {})).get("compilation_metadata")
        or {}
    )
    extracted_args = (
        launch.get("extracted_args")
        or (launch.get("payload", {})).get("extracted_args")
        or {}
    )

    # compile metadata subset we care about
    compile_block = {
        "num_warps": comp_meta.get("num_warps"),
        "num_stages": comp_meta.get("num_stages"),
        "arch": comp_meta.get("arch"),
        "backend": comp_meta.get("backend_name") or comp_meta.get("backend"),
        "triton_version": comp_meta.get("triton_version"),
        "hash": comp_meta.get("hash"),
    }

    # kwargs: include constexpr + explicit scalars used for launch (skip tensor args)
    kwargs = {}
    for k, v in extracted_args.items():
        val = _decode_arg(v)
        if val is _SKIP:
            continue
        kwargs[k] = val

    # tensor args: only tensors
    tensor_args = {
        k: v
        for k, v in extracted_args.items()
        if isinstance(v, dict) and v.get("type") == "tensor"
    }

    bundle = {
        "kernel_info": kernel_info,
        "compile": compile_block,
        "launch": {
            "grid": grid,
            "kwargs": kwargs,
        },
        "args": _pack_args(extracted_args),
        "tensor_args": _pack_args(tensor_args),
    }
    return bundle


def find_launch_index_from_line(ndjson_path: str, target_line: int) -> int:
    """Find the launch_index for a given line number in an NDJSON file."""
    if target_line <= 0:
        raise ValueError("Line numbers must be 1-based and positive.")

    launch_count = 0
    with open(ndjson_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i > target_line:
                # We've passed the target line without finding a launch event on it
                break
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("event_type") == "launch":
                    if i == target_line:
                        logger.debug(
                            "Found target launch event on line %d, launch_index is %d.",
                            target_line,
                            launch_count,
                        )
                        return launch_count
                    launch_count += 1
            except json.JSONDecodeError:
                # Warning already logged in _iter_events if it were used, but not here.
                logger.warning("Skipping malformed JSON line: %s", line)
                continue

    logger.error(
        "Could not find a 'launch' event on line %d in '%s'",
        target_line,
        ndjson_path,
    )
    raise ValueError(
        f"Could not find a 'launch' event on line {target_line} in {ndjson_path}"
    )
