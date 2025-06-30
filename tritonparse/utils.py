#  Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

# argument parser for OSS
parser = None

from .common import copy_local_to_tmpdir, is_fbcode, parse_logs, RankConfig, save_logs
from .source_type import Source, SourceType


def init_parser():
    global parser

    parser = argparse.ArgumentParser(
        description="analyze triton structured logs", conflict_handler="resolve"
    )

    # Add arguments for the parse command
    parser.add_argument(
        "source",
        help="Source of torch logs to be analyzed. It is expected to path to a local directory or log",
    )
    parser.add_argument(
        "-o",
        "--out",
        help="Output directory.",
        type=str,
    )
    parser.add_argument(
        "--overwrite",
        help="Delete out directory if it already exists. Only does something if --out is set",
        action="store_true",
    )
    parser.add_argument("-r", "--rank", help="Rank of logs to be analyzed", type=int)
    parser.add_argument(
        "--all-ranks",
        help="Analyze all ranks",
        action="store_true",
    )
    parser.add_argument(
        "-v", "--verbose", help="Verbose logging", action="store_true"
    )
    if is_fbcode():
        from tritonparse.fb.utils import append_parser

        append_parser(parser)
    return parser


def oss_parse(
    source: Optional[str] = None,
    out: Optional[str] = None,
    overwrite: bool = True,
    rank: Optional[int] = None,
    all_ranks: bool = False,
    verbose: bool = False,
):
    """
    Main function for the parse subcommand. It is for OSS only.

    Args:
        args: Command-line arguments

    Returns:
        Exit code
    """
    source = Source(source, verbose)
    rank_config = RankConfig.from_cli_args(rank, all_ranks, source.type)

    # Check output directory early if specified
    if out is not None:
        out_dir = Path(out)
        if out_dir.exists():
            if not overwrite:
                raise RuntimeError(
                    f"{out_dir} already exists, pass --overwrite to overwrite"
                )
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    # For signpost logging (not implemented in Python version)

    if source.type == SourceType.LOCAL:
        local_path = source.value
        # Copy the results to a temp directory, then parse them
        logs = copy_local_to_tmpdir(local_path, verbose)

    elif source.type == SourceType.LOCAL_FILE:
        local_path = source.value
        # Copy the single file to a temp directory, then parse it
        logs = copy_local_to_tmpdir(local_path, verbose)

    parsed_log_dir, _ = parse_logs(logs, rank_config, verbose)
    if out is not None:
        save_logs(Path(out), parsed_log_dir, overwrite, verbose)

def unified_parse_from_cli():
    parser = init_parser()
    args = parser.parse_args()
    return unified_parse(**vars(args))

def unified_parse(
    source: Optional[str] = None,
    out: Optional[str] = None,
    overwrite: bool = True,
    rank: Optional[int] = None,
    all_ranks: bool = False,
    verbose: bool = False,
    **kwargs,
):
    """
    Unified parse function that provides a flexible interface for parsing triton logs.

    Args:
        source: Input directory containing logs to parse. If None, will parse from command line arguments
        out: Output directory for parsed results. If None, results won't be saved to a specific location
        overwrite: Whether to overwrite existing output directory
        rank: Specific rank to analyze
        all_ranks: Whether to analyze all ranks
        verbose: Whether to enable verbose logging
    """
    # Choose the appropriate parse function
    if is_fbcode():
        from tritonparse.fb.utils import fb_parse as parse
    else:
        parse = oss_parse

    parse(
        source=source,
        out=out,
        overwrite=overwrite,
        rank=rank,
        all_ranks=all_ranks,
        verbose=verbose,
        **kwargs,
    )
