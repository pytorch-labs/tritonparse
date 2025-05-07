#!/usr/bin/env python3
"""
Extract source code mappings from Triton trace files and update the original JSON.
This script reads a JSON trace file containing Triton IR (TTIR, TTGIR) and PTX(AMDGCN),
and extracts bidirectional mappings between:
- Python ↔ TTIR
- Python ↔ TTGIR
- Python ↔ PTX(AMDGCN)
- TTIR ↔ TTGIR
- TTIR ↔ PTX(AMDGCN)
- TTGIR ↔ PTX(AMDGCN)
"""

import argparse
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

# the definition of the #loc directive. they are in the bottom of the IR files
# Example:#loc2 = loc("/tmp/torchinductor_yhao/yp/abcdef.py":20:28)
LOC_PATTERN = re.compile(r'#loc(\d*) = loc\("([^"]+)":(\d+):(\d+)\)')

# the reference to the #loc directive. they are in the end of lines of the IR files
# Example: loc(#loc2)
CODE_LOC_PATTERN = re.compile(r".*loc\(#loc(\d*)\)\s*$")

# this pattern is used in the first function arguments line.
DIRECT_FILE_PATTERN = re.compile(r'.*loc\("([^"]+)":(\d+):(\d+)\)')


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


def create_python_mapping(
    ir_maps: List[Tuple[str, Dict[str, Dict[str, Any]]]],
) -> Dict[int, Dict[str, List[int]]]:
    """
    Create a mapping from Python source code to IR mappings. We assume there is only one Python source code for each triton kernel.
    Args:
        ir_maps: A list of tuples containing the IR type and the IR mappings.

    Returns:
        A dictionary mapping Python source code line numbers to their corresponding IR mappings.
    """
    py_map = defaultdict(defaultdict(list))
    for ir_type, ir_map in ir_maps:
        for line_number, info in ir_map.items():
            py_line_number: int = info["line"]
            py_map[py_line_number][f"{ir_type}_lines"].append(line_number)
    return py_map


def extract_loc_definitions(ir_content: str) -> Dict[str, Dict[str, Any]]:
    """
    Extracts location definitions from the given IR content.

    This function searches for #loc directives in the provided IR content string.
    It identifies the main #loc directive, which is a special case located at the top
    of the IR files, and any subsequent #loc directives that define source file locations.

    Args:
        ir_content (str): The content of the IR file as a string.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping location IDs to their corresponding
        file names, line numbers, and column numbers.
    """
    locations = {}
    # The first #loc directive is a special case. It locates at the top of the IR files
    main_match = re.search(r'#loc = loc\("([^"]+)":(\d+):(\d+)\)', ir_content)
    if main_match:
        locations["1"] = {
            "file": main_match.group(1),
            "line": int(main_match.group(2)),
            "column": int(main_match.group(3)),
        }
    # #loc1 = loc(unknown) is another special case. We ignore it.
    for loc_id, filename, line, col in LOC_PATTERN.findall(ir_content):
        key = loc_id
        locations[key] = {"file": filename, "line": int(line), "column": int(col)}
    return locations


def extract_code_locations(ir_content: str) -> Dict[int, str]:
    """
    Extracts code location mappings from the given IR content.

    This function scans through the provided IR content line by line and identifies
    lines that contain location references. It uses regular expressions to match
    both the #loc directives and direct file references. The function returns a
    dictionary mapping line numbers to their corresponding location identifiers.
    Limitations:
        For the first function arguments line, it may use some #loc(file:line:col), DIRECT_FILE_PATTERN, we only use the first location reference.
    Args:
        ir_content (str): The content of the IR file as a string.

    Returns:
        Dict[int, str]: A dictionary mapping line numbers to location identifiers,
        which can be either a #loc identifier or a direct file reference.
    """
    line_to_loc = {}
    for i, line in enumerate(ir_content.split("\n"), start=1):
        if m := CODE_LOC_PATTERN.search(line):
            line_to_loc[i] = m.group(1) or "0"
        elif m := DIRECT_FILE_PATTERN.search(line):
            file_path, ln, col = m.groups()
            line_to_loc[i] = f"direct:{file_path}:{ln}:{col}"
    return line_to_loc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract source code mappings from Triton trace files."
    )
    parser.add_argument("-i", "--input", help="Path to the Triton trace NDJSON file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save the output files. If not specified, the input file's directory will be used.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output NDJSON path. If it is None, the default output file name will be set to {input}_mapped.ndjson in the parse function.",
    )
    return parser.parse_args()
