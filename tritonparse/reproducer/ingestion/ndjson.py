import json
from pathlib import Path
from typing import Dict, Any, List, Optional

def _iter_events(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # skip malformed lines
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
    return idx

def _get_launches(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in events if e.get("event_type") == "launch"]

def _resolve_kernel_source(launch: Dict[str, Any], comp_idx: Dict[str, Dict[str, Any]]) -> str:
    # In new format, launch has top-level compilation_metadata, not payload.*
    comp_meta = launch.get("compilation_metadata") or launch.get("payload", {}).get("compilation_metadata") or {}
    h = comp_meta.get("hash")
    if not h:
        return ""
    comp = comp_idx.get(h, {})
    payload = comp.get("payload") or {}
    py = payload.get("python_source") or {}
    return py.get("code", "")

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
                "is_contiguous": v.get("is_contiguous") if isinstance(v, dict) else None,
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

def build_context_bundle(ndjson_path: str, launch_index: int = 0) -> Dict[str, Any]:
    events = list(_iter_events(ndjson_path))
    launches = _get_launches(events)
    if not launches:
        raise RuntimeError("No launch events found in NDJSON.")
    if launch_index < 0 or launch_index >= len(launches):
        raise IndexError(f"launch_index out of range: {launch_index} (total {len(launches)})")
    launch = launches[launch_index]
    comp_idx = _index_compilations(events)
    kernel_source = _resolve_kernel_source(launch, comp_idx)

    # flatten launch fields (support both formats)
    grid = launch.get("grid") or (launch.get("payload", {})).get("grid")
    comp_meta = launch.get("compilation_metadata") or (launch.get("payload", {})).get("compilation_metadata") or {}
    extracted_args = launch.get("extracted_args") or (launch.get("payload", {})).get("extracted_args") or {}

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
        if isinstance(v, dict) and v.get("type") == "tensor":
            continue
        # pick usable value
        if isinstance(v, dict):
            val = v.get("value", v.get("repr"))
        else:
            val = v
        kwargs[k] = val

    # tensor args: only tensors
    tensor_args = {k: v for k, v in extracted_args.items() if isinstance(v, dict) and v.get("type") == "tensor"}

    bundle = {
        "kernel_source": kernel_source,
        "compile": compile_block,
        "launch": {
            "grid": grid,
            "kwargs": kwargs,
        },
        "args": _pack_args(extracted_args),
        "tensor_args": _pack_args(tensor_args),
    }
    return bundle


