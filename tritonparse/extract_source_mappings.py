#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.

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
import copy
import gzip
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SourceMapping")


# the definition of the #loc directive. they are in the bottom of the IR files
# Example:#loc2 = loc("/tmp/torchinductor_yhao/yp/abcdef.py":20:28)
LOC_PATTERN = re.compile(r'#loc(\d*) = loc\("([^"]+)":(\d+):(\d+)\)')

# the reference to the #loc directive. they are in the end of lines of the IR files
# Example: loc(#loc2)
CODE_LOC_PATTERN = re.compile(r".*loc\(#loc(\d*)\)\s*$")

# this pattern is used in the first function arguments line.
DIRECT_FILE_PATTERN = re.compile(r'.*loc\("([^"]+)":(\d+):(\d+)\)')


# the definition of the PTX loc directive.
# Example: .loc 1 0 50 // abcdef.py:0:50
PTX_LOC_PATTERN = re.compile(
    r"^\s*\.loc\s+\d+\s+(\d+)\s+(\d+)\s+//\s*(.+?):(\d+):(\d+)"
)

# the definition of the AMDGCN loc directive.
# Example: .loc	1 32 30                         ; abcd.py:32:30
# .loc	1 32 46 is_stmt 0               ; abcd.py:32:46
AMDGCN_LOC_PATTERN = re.compile(
    r".*loc\s+(\d+)\s+(\d+)\s+(\d+)(?:\s+[^;]*)?;\s*(.+?):(\d+):(\d+)"
)


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
    py_map = defaultdict(lambda: defaultdict(list))
    for ir_type, ir_map in ir_maps:
        for line_number, info in ir_map.items():
            py_line_number: int = info["line"]
            py_map[py_line_number][f"{ir_type}_lines"].append(line_number)
    return {k: dict(v) for k, v in py_map.items()}


def create_ir_mapping(
    source_map: Dict[str, Dict[str, Any]], target_map: Dict[str, Dict[str, Any]]
) -> Dict[str, List[int]]:
    """
    Create a mapping from source IR lines to target IR lines.

    This function takes two mappings: one for source IR and one for target IR, and creates a new mapping
    that associates lines in the source IR with corresponding lines in the target IR based on their file,
    line, and column information.

    Args:
        source_map (Dict[str, Dict[str, Any]]): A dictionary mapping source IR line numbers to their source file,
            line, and column information.
        target_map (Dict[str, Dict[str, Any]]): A dictionary mapping target IR line numbers to their source file,
            line, and column information.

    Returns:
        Dict[str, List[int]]: A dictionary mapping source IR line numbers to lists of corresponding target IR line numbers.
    """
    source_to_target = defaultdict(list)

    # Build a mapping from source file locations to target lines
    for tgt_line, tgt_info in target_map.items():
        if "file" in tgt_info and "line" in tgt_info:
            key = f"{tgt_info['file']}:{tgt_info['line']}:{tgt_info.get('column', 0)}"
            source_to_target[key].append(int(tgt_line))

    # Map source lines to target lines
    mapping = {}
    for src_line, src_info in source_map.items():
        if "file" in src_info and "line" in src_info:
            key = f"{src_info['file']}:{src_info['line']}:{src_info.get('column', 0)}"
            if key in source_to_target:
                mapping[src_line] = sorted(source_to_target[key])

    return mapping


def create_bidirectional_mapping(
    source_map: Dict[str, Dict[str, Any]],
    target_map: Dict[str, Dict[str, Any]],
    source_type: str,
    target_type: str,
) -> None:
    """
    Create bidirectional mappings between two IR types and update their mapping dictionaries.

    This function creates mappings from source IR to target IR and vice versa, then
    updates both mapping dictionaries with the line references.

    Args:
        source_map: Dictionary mapping source IR line numbers to source locations
        target_map: Dictionary mapping target IR line numbers to source locations
        source_type: String identifier for the source IR type (e.g., 'ttir', 'ttgir', 'ptx')
        target_type: String identifier for the target IR type (e.g., 'ttir', 'ttgir', 'ptx')
    """
    # Create forward mapping (source to target)
    source_to_target = create_ir_mapping(source_map, target_map)

    # Add target line references to source mappings
    for source_line, target_lines in source_to_target.items():
        if source_line in source_map and target_lines:
            source_map[source_line][f"{target_type}_lines"] = target_lines

    # Create reverse mapping (target to source)
    target_to_source = create_ir_mapping(target_map, source_map)

    # Add source line references to target mappings
    for target_line, source_lines in target_to_source.items():
        if target_line in target_map:
            target_map[target_line][f"{source_type}_lines"] = source_lines

    logger.debug(f"Created {source_type} to {target_type} mappings (and reverse)")


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


def extract_ptx_amdgcn_mappings(
    content: str, other_mappings: List[Any] = None, ir_type: str = "ptx"
) -> Dict[str, Dict[str, Any]]:
    """
    Extract mappings from PTX code where `.loc` directives provide source file and line info.
    This function only processes code between the function begin and end markers (e.g., "// -- Begin function" and "// -- End function"). The PTX source code line mapping is quite different from that of other IRs. It segments the PTX code using the .loc directive, where each .loc directive provides information for mapping to a source code line.

    This function:
    1. Identifies the function boundary in PTX code
    2. Only processes code within the function boundary
    3. Maps PTX lines with `.loc` directives to source files and line numbers
    4. Associates subsequent code lines with the most recent `.loc` directive

    Args:
        ptx_content: The content of the PTX file

    Returns:
        Dictionary mapping PTX line numbers to source location information
    """
    mappings = {}
    current_mapping = None

    # Mark function scope
    function_start_line = 0
    function_end_line = 0
    # filename: {file_path, ...}
    referenced_files = defaultdict(set)
    if other_mappings is None:
        other_mappings = []
    for other in other_mappings:
        for _, info in other.items():
            if "file" in info:
                file_name = os.path.basename(info["file"])
                referenced_files[file_name].add(info["file"])

    def get_file_path(filename: str) -> str:
        file_path = filename
        if not os.path.isabs(filename):
            logger.debug(
                f"Filename '{filename}' does not contain a path. Attempting to resolve."
            )
            # Attempt to resolve the filename to a full path using referenced_files
            if filename in referenced_files:
                if len(referenced_files[filename]) > 1:
                    logger.debug(
                        f"Filename '{filename}' has multiple file paths. Using the first one."
                    )
                file_path = list(referenced_files[filename])[0]
                logger.debug(f"Resolved filename '{filename}' to {file_path}")
            else:
                logger.debug(f"Filename '{filename}' not found in referenced files.")
        return file_path

    # Regular expressions to match function start and end markers
    # @TODO: need to double check if the PTX content only contains one function
    begin_func_pattern = re.compile(
        r"(?:(?://|;)\s*(?:\.globl\s+\S+\s+)?|\.globl\s+\S+\s+;\s*)--\s*Begin function"
    )
    end_func_pattern = re.compile(r"(?://|;)\s*--\s*End function")

    # First scan: find function boundaries
    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        if begin_func_pattern.search(line) and function_start_line == 0:
            function_start_line = i
        elif end_func_pattern.search(line) and function_start_line > 0:
            function_end_line = i
            break

    # If no function boundaries are found, return empty mapping
    if function_start_line == 0 or function_end_line == 0:
        logger.warning(
            f"Could not identify {ir_type} function boundaries. No {ir_type} mappings generated."
        )
        return mappings

    logger.debug(
        f"Processing {ir_type} function from line {function_start_line} to {function_end_line}"
    )

    is_ptx = ir_type == "ptx"
    is_amdgcn = ir_type == "amdgcn"

    tmp_loc_pattern = PTX_LOC_PATTERN if is_ptx else AMDGCN_LOC_PATTERN
    # Second scan: process code within function body
    # pay attention to the line number, it starts from 0 but the function_start_line starts from 1
    for i, line in enumerate(
        lines[function_start_line:function_end_line], start=function_start_line + 1
    ):
        try:
            # Check .loc directive line
            match = tmp_loc_pattern.match(line)
            if match:
                if is_ptx:
                    py_line, py_col, filename, _, _ = match.groups()
                elif is_amdgcn:
                    py_file_index, py_line, py_col, filename, _, _ = match.groups()
                else:
                    logger.error(f"Unknown IR type: {ir_type}")
                    raise ValueError(f"Unknown IR type: {ir_type}")
                file_path = get_file_path(filename)
                # Create new mapping
                current_mapping = {
                    "file": file_path,
                    "line": int(py_line),
                    "column": int(py_col),
                    f"{ir_type}_line": i,
                }
                # Store mapping
                mappings[str(i)] = current_mapping
            elif current_mapping:
                # For lines without their own .loc after .loc directive, associate with the nearest .loc mapping
                # Only process non-empty, non-comment meaningful code lines
                line_content = line.strip()
                if line_content and not (
                    (is_ptx and line_content.startswith("//"))
                    or (is_amdgcn and line_content.startswith(";"))
                ):
                    mappings[str(i)] = {
                        "file": current_mapping["file"],
                        "line": current_mapping["line"],
                        "column": current_mapping["column"],
                        f"{ir_type}_line": i,
                    }
        except Exception as e:
            logger.error(f"Error processing line {i}: {e}")
            logger.error(f"Line content: {line}")
            raise e
    return mappings


def generate_source_mappings(
    ir_content: str, ir_type: str, other_mappings: List[Any] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Generate source mappings from intermediate representation (IR) content to the source file.
    Example:
    loc definition: Line 39 in ttir: #loc2 = loc("/tmp/torchinductor_yhao/yp/abcdef.py":20:28)
    loc reference: Line 9 in ttir: %0 = tt.get_program_id x : i32 loc(#loc2)
    Then, the output will be:
    {
        "9": {
            "file": "/tmp/torchinductor_yhao/yp/abcdef.py",
            "line": 20,
            "column": 28,
            "ttir_line": 9
        },
    }

    Args:
        ir_content (str): The content of the intermediate representation.
        ir_type (str): The type of the intermediate representation (e.g., 'ttir').
        other_mappings (List[Any]): A collection of additional mappings, primarily utilized for PTX mappings since PTX's location annotations reference the file name instead of the complete path.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping line numbers to their corresponding source file,
        line, column, and the line number in the IR.
    """
    if other_mappings is None:
        other_mappings = []
    if ir_type == "ptx" or ir_type == "amdgcn":
        return extract_ptx_amdgcn_mappings(ir_content, other_mappings, ir_type)

    loc_defs = extract_loc_definitions(ir_content)
    logger.debug(f"Found {len(loc_defs)} #loc definitions")

    loc_refs = extract_code_locations(ir_content)
    logger.debug(f"Found {len(loc_refs)} loc references")

    mappings = {}
    for ln, loc_id in loc_refs.items():
        if loc_id.startswith("direct:"):
            _, file_path, line, col = loc_id.split(":", 3)
            mappings[str(ln)] = {
                "file": file_path,
                "line": int(line),
                "column": int(col),
                f"{ir_type}_line": ln,
            }
        elif loc_id in loc_defs:
            info = loc_defs[loc_id]
            mappings[str(ln)] = {
                "file": info["file"],
                "line": info["line"],
                "column": info["column"],
                f"{ir_type}_line": ln,
            }

    return mappings


def process_ir(
    key: str,
    file_content: Dict[str, str],
    file_path: Dict[str, str],
    other_mappings: List[Any] = None,
):
    # Generate source mappings for each IR type
    # the key should be the full file name with extension for the IR files
    if not key:
        return {}
    logger.debug(f"Processing {key}")
    ir_content = file_content.get(key, None)
    if not ir_content:
        ir_file_path = file_path.get(key, None)
        if not ir_file_path:
            logger.warning(f"No content found for {key}")
            return {}
        with open(ir_file_path, "r") as f:
            ir_content = f.read()
    mapping = generate_source_mappings(ir_content, key.split(".")[1], other_mappings)
    logger.debug(f"Generated source mapping for {key}")
    return mapping


def parse_single_trace_content(trace_content: str) -> str:
    """
    Process a single trace content and extract source code mappings.

    This function takes a trace content as input, extracts the IR files, generates source mappings,
    creates bidirectional mappings between different IR types, and updates the payload with the mappings.

    Args:
        trace_content (str): The content of the trace file as a string.

    Returns:
        str: The updated trace content with source mappings as a JSON string.
    """

    entry = json.loads(trace_content)
    if entry.get("event_type") == "compilation":
        payload = entry.setdefault("payload", {})
        file_content = payload.get("file_content", {})
        file_path = payload.get("file_path", {})

        # Find the IR file keys
        ttir_key = next((k for k in file_content if k.endswith(".ttir")), None)
        ttgir_key = next((k for k in file_content if k.endswith(".ttgir")), None)
        ptx_key = next((k for k in file_content if k.endswith(".ptx")), None)
        amdgcn_key = next((k for k in file_content if k.endswith(".amdgcn")), None)
        # Skip if no IR files found
        if not (ttir_key or ttgir_key or ptx_key or amdgcn_key):
            logger.warning("No IR files found in the payload.")
            return trace_content

        # generate ttir->source, ttgir->source, ptx->source
        ttir_map = process_ir(ttir_key, file_content, file_path)
        ttgir_map = process_ir(ttgir_key, file_content, file_path)
        ptx_map = process_ir(ptx_key, file_content, file_path, [ttir_map, ttgir_map])
        amdgcn_map = process_ir(
            amdgcn_key, file_content, file_path, [ttir_map, ttgir_map]
        )

        # Create bidirectional mappings between all IR types
        ir_maps = {
            "ttir": ttir_map,
            "ttgir": ttgir_map,
            "ptx": ptx_map,
            "amdgcn": amdgcn_map,
        }

        # Create mappings between all pairs of IR types
        ir_types = list(ir_maps.keys())
        for i, src_type in enumerate(ir_types):
            for tgt_type in ir_types[i + 1 :]:
                if ir_maps[src_type] and ir_maps[tgt_type]:
                    create_bidirectional_mapping(
                        ir_maps[src_type], ir_maps[tgt_type], src_type, tgt_type
                    )
                    logger.debug(
                        f"Created bidirectional mapping between {src_type} and {tgt_type}"
                    )

        py_map = {}

        if "python_source" in payload:
            logger.debug(
                f"Added Python source information (lines {payload['python_source']['start_line']}-{payload['python_source']['end_line']})"
            )

            # 4. Create Python source to IR mappings. We use the original line numbers as key in the python source code.
            # Create a list of valid IR mappings, filtering out None keys
            ir_mappings = []
            ir_keys_and_maps = [
                (ttir_key, ttir_map),
                (ttgir_key, ttgir_map),
                (ptx_key, ptx_map),
                (amdgcn_key, amdgcn_map),
            ]

            for key, mapping in ir_keys_and_maps:
                if key:
                    ir_mappings.append((get_file_extension(key), mapping))

            py_map = create_python_mapping(ir_mappings)

        # Store the mappings in the payload
        payload["source_mappings"] = {
            "ttir": ttir_map,
            "ttgir": ttgir_map,
            **({"ptx": ptx_map} if ptx_map else {}),
            **({"amdgcn": amdgcn_map} if amdgcn_map else {}),
            "python": py_map,
        }
    # NDJSON format requires a newline at the end of each line
    return json.dumps(entry, separators=(",", ":")) + "\n"


def parse_single_file(
    file_path: str,
    output_dir: str = None,
    split_by_frame_id_and_compile_id: bool = True,
):
    """
    Process a single file and extract source code mappings.

    This function takes a file path as input, reads the file content, and extracts
    source code mappings from Triton trace JSON files. It processes each line of the file,
    parses the trace content to extract IR mappings, and writes the updated content
    to output files.

    Args:
        file_path (str): The path to the file to be processed.
        output_dir (str, optional): Directory to save the output files with mappings.
        split_by_frame_id_and_compile_id (bool, optional): Whether to split output files
            by frame_id and compile_id. Defaults to True.

    Returns:
        None. The function writes the processed data to files in the output_dir.
        Each output file will contain the original trace data enriched with source mappings
        in NDJSON format (one JSON object per line).
    """
    outputs = defaultdict(list)
    # Dictionary to track kernel hash to output file mapping
    # Format: {hash: output_file_path}
    kernel_hash_to_file = {}

    # Set default output directory if not provided
    output_dir = output_dir or os.path.dirname(file_path)

    # Check if input file is compressed based on file extension
    is_compressed_input = file_path.endswith(".bin.ndjson")

    # Open file in appropriate mode - use gzip.open for compressed files
    if is_compressed_input:
        # Use gzip.open which automatically handles member concatenation
        file_handle = gzip.open(file_path, "rt", encoding="utf-8")
    else:
        file_handle = open(file_path, "r")

    with file_handle as f:
        file_name = os.path.basename(file_path)
        # Handle .bin.ndjson extension properly
        if is_compressed_input:
            file_name_without_extension = file_name[:-11]  # Remove .bin.ndjson
        else:
            file_name_without_extension = os.path.splitext(file_name)[0]

        # Process lines uniformly for both compressed and uncompressed files
        for i, line in enumerate(f):
            logger.debug(f"Processing line {i + 1} in {file_path}")

            json_str = line.strip()
            if not json_str:
                continue

            parsed_line = parse_single_trace_content(json_str)
            if not parsed_line:
                logger.warning(f"Failed to parse line {i + 1} in {file_path}")
                continue

            parsed_json = json.loads(parsed_line)
            event_type = parsed_json.get("event_type", None)
            payload = parsed_json.get("payload", None)

            output_file = None

            if event_type == "compilation":
                # Handle compilation events
                if split_by_frame_id_and_compile_id:
                    if not payload:
                        logger.warning("No payload found in the compilation event.")
                        continue
                    pt_info = payload.get("pt_info", {})
                    frame_id = pt_info.get("frame_id", None)
                    frame_compile_id = pt_info.get("frame_compile_id", None)
                    compiled_autograd_id = pt_info.get("compiled_autograd_id", "-")
                    attempt_id = pt_info.get("attempt_id", 0)
                    output_file_name = ""
                    if frame_id is not None or frame_compile_id is not None:
                        output_file_name = f"f{frame_id}_fc{frame_compile_id}_a{attempt_id}_cai{compiled_autograd_id}.ndjson"
                    else:
                        logger.debug(
                            "No frame_id or frame_compile_id found in the payload."
                        )
                        output_file_name = (
                            f"{file_name_without_extension}_mapped.ndjson"
                        )
                else:
                    output_file_name = f"{file_name_without_extension}_mapped.ndjson"
                output_file = os.path.join(output_dir, output_file_name)

                # Record the kernel hash to file mapping
                if payload:
                    metadata = payload.get("metadata", {})
                    kernel_hash = metadata.get("hash", None)
                    if kernel_hash:
                        kernel_hash_to_file[kernel_hash] = output_file
                        logger.debug(
                            f"Recorded compilation hash {kernel_hash} -> {output_file}"
                        )

            elif event_type == "launch":
                # Handle launch events
                compilation_metadata = parsed_json.get("compilation_metadata", {})
                kernel_hash = compilation_metadata.get("hash", None)
                if kernel_hash:
                    if kernel_hash in kernel_hash_to_file:
                        # Use the same output file as the corresponding compilation event
                        output_file = kernel_hash_to_file[kernel_hash]
                        logger.debug(
                            f"Found launch event for hash {kernel_hash}, using file {output_file}"
                        )
                    else:
                        logger.warning(
                            f"No compilation event found for launch hash {kernel_hash}, skipping this launch event."
                        )
                        continue
                else:
                    logger.warning(
                        f"Launch event without kernel hash, skipping this launch event in {file_path}:{i + 1}."
                    )

            else:
                logger.info(f"Skipping line {i + 1} with event type {event_type}")
            if output_file:
                outputs[output_file].append(parsed_line)
                logger.debug(f"output file: {output_file}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for output_file, parsed_lines in outputs.items():
        with open(output_file, "w") as out:
            out.writelines(parsed_lines)

    # Post-process to generate launch diffs for each kernel hash
    for output_file, parsed_lines in outputs.items():
        kernels_by_hash = defaultdict(lambda: {"compilation": None, "launches": []})

        # Group compilations and launches by kernel hash
        for line in parsed_lines:
            event = json.loads(line)
            event_type = event.get("event_type")
            if event_type == "compilation":
                kernel_hash = event.get("payload", {}).get("metadata", {}).get("hash")
                if kernel_hash:
                    kernels_by_hash[kernel_hash]["compilation"] = event
            elif event_type == "launch":
                kernel_hash = event.get("compilation_metadata", {}).get("hash")
                if kernel_hash:
                    kernels_by_hash[kernel_hash]["launches"].append(event)

        # Generate and append launch_diff event for each kernel
        for kernel_hash, data in kernels_by_hash.items():
            compilation_event = data["compilation"]
            launches = data["launches"]

            if compilation_event and launches:
                sames, diffs = _generate_launch_diff(launches)
                launch_diff_event = {
                    "event_type": "launch_diff",
                    "hash": kernel_hash,
                    "name": compilation_event.get("payload", {}).get("metadata", {}).get("name"),
                    "total_launches": len(launches),
                    "diffs": diffs,
                    "sames": sames,
                }
                # Append the new event to the list of lines for the current output file
                outputs[output_file].append(json.dumps(launch_diff_event, separators=(",", ":")) + "\n")

    # Final write to files
    for output_file, final_lines in outputs.items():
        with open(output_file, "w") as out:
            out.writelines(final_lines)


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


# Fields that are expected to vary but are not useful to list out in the diff.
SUMMARY_FIELDS = ["pid", "timestamp", "stream", "function", "data_ptr"]


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
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


def _generate_launch_diff(launches: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compares a list of launch events and returns dictionaries of same and different fields.
    """
    if not launches:
        return {}, {}

    # Use the first launch as the base for comparison
    base_launch_flat = _flatten_dict(launches[0])
    sames = copy.deepcopy(base_launch_flat)
    diffs = defaultdict(lambda: defaultdict(list))

    for i, launch in enumerate(launches):
        launch_flat = _flatten_dict(launch)
        for key, value in launch_flat.items():
            if key not in sames:
                # This key is already known to be different
                # Check if the value is new
                if value not in [v["value"] for v in diffs[key]["values"]]:
                    diffs[key]["values"].append({"value": value, "first_occurrence_index": i})
                continue

            if sames[key] != value:
                # A field that was the same until now is different
                base_value = sames.pop(key)
                diffs[key]["values"].append({"value": base_value, "first_occurrence_index": 0})
                diffs[key]["values"].append({"value": value, "first_occurrence_index": i})

    # Finalize the structure of diffs
    final_diffs = {}
    for key, data in diffs.items():
        is_summary = any(summary_key in key for summary_key in SUMMARY_FIELDS)
        unique_values = data["values"]
        if is_summary:
            final_diffs[key] = {
                "type": "summary",
                "unique_count": len(unique_values),
                "summary_text": f"Varies across {len(unique_values)} unique values",
            }
        else:
            final_diffs[key] = {
                "type": "values_list",
                "values": unique_values,
            }

    # Unflatten the sames dictionary to restore its original structure
    sames_unflattened = _unflatten_dict(sames)

    return sames_unflattened, final_diffs


if __name__ == "__main__":
    args = parse_args()
    if args.input:
        parse_single_file(args.input, args.output_dir)
    else:
        logger.error("No input file specified.")
