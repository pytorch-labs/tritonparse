from typing import Any, Dict, List


def get_file_extension(filename: str) -> str:
    """
    Get the file extension from a given filename or return the filename itself if it has no extension.

    Args:
        filename (str): The filename or file extension.

    Returns:
        str: The file extension or the filename itself if no extension is present.
    """
    # Split the filename by '.' and return the last part if it exists
    parts = filename.split(".")
    return parts[-1] if len(parts) > 1 else filename


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flattens a nested dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """
    Unflattens a dictionary with delimited keys.
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        d_ref = result
        for part in parts[:-1]:
            if part not in d_ref:
                d_ref[part] = {}
            d_ref = d_ref[part]
        d_ref[parts[-1]] = value
    return result


def _to_ranges(indices: List[int]) -> List[Dict[str, int]]:
    """
    Converts a sorted list of indices into a list of continuous ranges.
    e.g., [0, 1, 2, 5, 6, 8] -> [{'start': 0, 'end': 2}, {'start': 5, 'end': 6}, {'start': 8, 'end': 8}]
    """
    if not indices:
        return []

    indices = sorted(indices)
    ranges = []
    start = indices[0]
    end = indices[0]

    for i in range(1, len(indices)):
        if indices[i] == end + 1:
            end = indices[i]
        else:
            ranges.append({"start": start, "end": end})
            start = end = indices[i]

    ranges.append({"start": start, "end": end})
    return ranges


def _is_autotune_benchmark_launch(stack: List[Dict[str, Any]]) -> bool:
    """Checks if a stack trace corresponds to an autotune benchmark launch."""
    for frame in stack:
        filename = frame.get("filename", "")
        func_name = frame.get("name")
        if "triton/runtime/autotuner.py" in filename and func_name == "_bench":
            return True
    return False


def get_autotune_session_id(stack_trace: List[Dict[str, Any]]) -> str | None:
    """
    Generate a session ID for an autotune instance by finding the user-level
    call site.

    Args:
        stack_trace: The stack trace from a compilation or launch event.

    Returns:
        A unique session ID string (e.g., "filename:lineno:func_name") if the
        event is part of an autotune process, otherwise None.
    """
    try:
        # Reverse the stack to search from the most recent call upwards
        reversed_stack = reversed(stack_trace)

        # Find the first frame corresponding to the autotuner's `run` method
        for i, frame in enumerate(reversed_stack):
            filename = frame.get("filename", "")
            func_name = frame.get("name")
            if "autotuner.py" in filename and func_name in ["run", "_bench"]:
                # The next frame in the original stack is the user's call site
                user_frame_index = len(stack_trace) - i - 2
                if user_frame_index >= 0:
                    user_frame = stack_trace[user_frame_index]
                    return (
                        f"{user_frame['filename']}:{user_frame['line']}:"
                        f"{user_frame['name']}"
                    )
    except (IndexError, KeyError):
        # Fallback if stack trace structure is unexpected
        return None
    return None
