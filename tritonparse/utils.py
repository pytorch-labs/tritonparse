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

    # For signpost logging (not implemented in Python version)

    if source.type == SourceType.LOCAL:
        local_path = source.value
        # Copy the results to a temp directory, then parse them
        logs = copy_local_to_tmpdir(local_path, verbose)

    elif source.type == SourceType.LOCAL_FILE:
        local_path = source.value

        if args.out is not None:
            out_dir = Path(args.out)
            if out_dir.exists():
                if not args.overwrite:
                    raise RuntimeError(
                        f"{out_dir} already exists, pass --overwrite to overwrite"
                    )
                shutil.rmtree(out_dir)

            os.makedirs(out_dir, exist_ok=True)
            return

    parsed_log_dir, _ = parse_logs(logs, rank_config, verbose)
    if args.out is not None:
        save_logs(Path(args.out), parsed_log_dir, args.overwrite, verbose)


def unified_parse(log_dir: str, output_dir: str = None, overwrite: bool = True, overwrite_manifold: bool = True, **kwargs):
    """
    Unified parse function that provides a flexible interface for parsing triton logs.

    Args:
        log_dir: Input directory containing logs to parse
        output_dir: Output directory for parsed results. If None, results won't be saved to a specific location
        overwrite: Whether to overwrite existing output directory
        overwrite_manifold: Whether to overwrite existing manifold output directory
        **kwargs: Additional arguments like rank, all_ranks, verbose, etc.
    """
    parser = init_parser()
    args = [log_dir]

    if output_dir:
        args.extend(["-o", output_dir])

    if overwrite:
        args.append("--overwrite")

    # Handle additional kwargs
    if kwargs.get("rank", None):
        args.extend(["-r", kwargs.get("rank")])
    if kwargs.get("all_ranks", False):
        args.append("--all-ranks")
    if kwargs.get("verbose", False):
        args.append("--verbose")

    if is_fbcode():
        from tritonparse.fb.utils import fb_parse as parse
        if overwrite_manifold:
            args.append("--overwrite-manifold")
    else:
        parse = oss_parse

    parse(parser.parse_args(args))
