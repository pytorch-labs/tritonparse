#!/usr/bin/env python3
"""
Script to compare two records from ndjson files.

Usage:
    # Compare records from two different files
    python compare_two_record.py file1.ndjson file2.ndjson --line1 INDEX1 --line2 INDEX2
    
    # Compare two records from the same file
    python compare_two_record.py file.ndjson --line1 INDEX1 --line2 INDEX2
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Set, Tuple, Union


def load_record(file_path: str, line_idx: int) -> Dict[str, Any]:
    """
    Load a specific record from an ndjson file by line index (0-based).
    
    Args:
        file_path: Path to the ndjson file
        line_idx: 0-based line index to load
        
    Returns:
        The parsed JSON object from the specified line
    """
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i == line_idx:
                    return json.loads(line)
            raise ValueError(f"Line index {line_idx} is out of range for file {file_path}")
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON at line {line_idx} in file {file_path}")
        sys.exit(1)


def compare_values(val1: Any, val2: Any, path: str = "") -> List[str]:
    """
    Compare two values and return a list of differences with their paths.
    
    Args:
        val1: First value to compare
        val2: Second value to compare
        path: Current path in the JSON structure
        
    Returns:
        List of differences with their paths
    """
    differences = []
    
    # If types are different
    if type(val1) != type(val2):
        differences.append(f"{path}: Type mismatch - {type(val1).__name__} vs {type(val2).__name__}")
        return differences
    
    # If dictionaries
    if isinstance(val1, dict):
        # Find keys in val1 but not in val2
        for key in val1:
            if key not in val2:
                differences.append(f"{path}.{key}: Key only in first record")
                continue
            
            # Recursively compare values
            differences.extend(compare_values(val1[key], val2[key], f"{path}.{key}" if path else key))
        
        # Find keys in val2 but not in val1
        for key in val2:
            if key not in val1:
                differences.append(f"{path}.{key}: Key only in second record")
    
    # If lists
    elif isinstance(val1, list):
        if len(val1) != len(val2):
            differences.append(f"{path}: List length mismatch - {len(val1)} vs {len(val2)}")
        
        # Compare elements
        for i in range(min(len(val1), len(val2))):
            differences.extend(compare_values(val1[i], val2[i], f"{path}[{i}]"))
    
    # If primitive values
    elif val1 != val2:
        # Special handling for floating point values in strings (like timestamps)
        if isinstance(val1, str) and isinstance(val2, str):
            if "%f" in val1 or "%f" in val2:
                # Skip comparison for strings containing %f placeholder
                return differences
        
        differences.append(f"{path}: Value mismatch - {val1} vs {val2}")
    
    return differences


def print_record_summary(record: Dict[str, Any], label: str) -> None:
    """Print a summary of the record."""
    print(f"\n{label} Summary:")
    print(f"  Event Type: {record.get('event_type', 'N/A')}")
    print(f"  Timestamp: {record.get('timestamp', 'N/A')}")
    print(f"  PID: {record.get('pid', 'N/A')}")
    
    # Print more specific details based on event_type if needed
    if record.get('event_type') == 'compilation':
        metadata = record.get('payload', {}).get('metadata', {})
        print(f"  Kernel Name: {metadata.get('name', 'N/A')}")
        print(f"  Backend: {metadata.get('backend_name', 'N/A')}")
        print(f"  Cache Hit: {metadata.get('cache_hit', 'N/A')}")
    elif record.get('event_type') == 'launch':
        print(f"  Kernel Name: {record.get('name', 'N/A')}")
        print(f"  Grid: {record.get('grid', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description='Compare two records from ndjson files.')
    parser.add_argument('file1', help='First ndjson file')
    parser.add_argument('file2', nargs='?', help='Second ndjson file (optional, defaults to file1)')
    parser.add_argument('--line1', type=int, required=True, help='Line index for first record (0-based)')
    parser.add_argument('--line2', type=int, help='Line index for second record (0-based, defaults to line1)')
    parser.add_argument('--ignore-keys', type=str, nargs='+', default=[], 
                        help='Keys to ignore during comparison (e.g., "timestamp" "stack")')
    parser.add_argument('--summary-only', action='store_true', help='Only show summary, not full differences')
    
    args = parser.parse_args()
    
    # If file2 is not provided, use file1
    file2 = args.file2 if args.file2 else args.file1
    
    # If line2 is not provided, use line1
    line2 = args.line2 if args.line2 is not None else args.line1
    
    # Load records
    record1 = load_record(args.file1, args.line1)
    record2 = load_record(file2, line2)
    
    # Print record summaries
    print_record_summary(record1, f"Record 1 ({args.file1}:{args.line1})")
    print_record_summary(record2, f"Record 2 ({file2}:{line2})")
    
    # Remove ignored keys
    for key in args.ignore_keys:
        if key in record1:
            del record1[key]
        if key in record2:
            del record2[key]
    
    # Compare records
    differences = compare_values(record1, record2)
    
    # Print results
    if not args.summary_only:
        if differences:
            print("\nDifferences:")
            for diff in differences:
                print(f"  {diff}")
        else:
            print("\nNo differences found between the records.")
    else:
        print(f"\nNumber of differences: {len(differences)}")
    
    # Return exit code based on whether differences were found
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
