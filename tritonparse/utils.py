#  Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import shutil
from pathlib import Path

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
        "-e", "--export", help="For export specific logs", action="store_true"
    )
    parser.add_argument("-v", "--verbose", help="Verbose logging", action="store_true")
    if is_fbcode():
        from tritonparse.fb.utils import append_parser

        append_parser(parser)
    return parser


def oss_parse(args):
    """
    Main function for the parse subcommand. It is for OSS only.

    Args:
        args: Command-line arguments

    Returns:
        Exit code
    """
    verbose = args.verbose
    source = Source(args.source, verbose)
    rank_config = RankConfig.from_cli_args(args.rank, args.all_ranks, source.type)

    # Check output directory early if specified
    if args.out is not None:
        out_dir = Path(args.out)
        if out_dir.exists():
            if not args.overwrite:
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
    if args.out is not None:
        save_logs(Path(args.out), parsed_log_dir, args.overwrite, verbose)


def unified_parse(args=None):
    """
    Unified parsing function that handles both fbcode and OSS environments.

    This function provides a single entry point for parsing triton logs,
    automatically selecting the appropriate parsing backend (fb_parse or oss_parse)
    based on the current environment.

    Args:
        args: Optional argument. Can be:
            - None: Will parse command line arguments automatically
            - str: Treated as parsed_log_dir path, will add --overwrite flag
            - argparse.Namespace: Pre-parsed arguments object

    Returns:
        None

    Raises:
        RuntimeError: If parsing fails or required arguments are missing
    """
    parser = init_parser()
    if args is None:
        args = parser.parse_args()
    elif isinstance(args, str):
        # If args is a string, treat it as parsed_log_dir
        args = parser.parse_args([args, "--overwrite"])

    if is_fbcode():
        from tritonparse.fb.utils import fb_parse as parse
    else:
        parse = oss_parse

    parse(args)
