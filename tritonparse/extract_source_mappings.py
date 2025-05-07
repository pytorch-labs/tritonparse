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
from collections import defaultdict
from typing import Any, Dict, List, Tuple


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
