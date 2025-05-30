import gzip

import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

from .extract_source_mappings import parse_single_file
from .tp_logger import logger


LOG_RANK_REGEX = re.compile(r"rank_(\d+)")
LOG_PREFIX = "dedicated_log_triton_trace_"


def is_fbcode():
    try:
        from .fb import IS_FBCODE_CHECK  # noqa

        return True
    except ImportError as e:
        print(e)
        return False


if is_fbcode():
    from .fb.source_type import SourceType
else:
    from .source_type import SourceType


class Rank:
    """Class representing a rank in distributed training."""

    def __init__(self, rank_value: Optional[int] = None):
        """
        Initialize a Rank object.

        Args:
            rank_value: Specific rank value, or None for default rank
        """
        if rank_value is not None:
            self.value = rank_value
            self.is_default = False
        else:
            self.value = 0
            self.is_default = True

    def to_string(self, prefix: str = "") -> str:
        """
        Convert rank to string representation with optional prefix.

        Args:
            prefix: Prefix to add before rank string

        Returns:
            String representation of the rank
        """
        if self.is_default:
            return ""
        return f"{prefix}rank_{self.value}"

    def to_int(self) -> int:
        """
        Convert rank to integer value.

        Returns:
            Integer value of the rank
        """
        return self.value


class RankConfig:
    """Configuration for handling ranks in log processing."""

    def __init__(
        self,
        rank: Optional[Rank] = None,
        all_ranks: bool = False,
        is_local: bool = False,
    ):
        """
        Initialize a RankConfig object.

        Args:
            rank: Specific rank to process
            all_ranks: Whether to process all ranks
            is_local: Whether processing local logs
        """
        self.rank = rank
        self.all_ranks = all_ranks
        self.is_local = is_local

    @classmethod
    def from_cli_args(
        cls, rank: Optional[int], all_ranks: bool, source_type: SourceType
    ) -> "RankConfig":
        """
        Create a RankConfig from command line arguments.

        Args:
            rank: Specific rank value from CLI
            all_ranks: Whether --all-ranks flag was specified
            source_type: Type of source

        Returns:
            Configured RankConfig object
        """
        if all_ranks:
            if rank is not None:
                raise ValueError("Can't specify both a rank and --all-ranks")
            return cls(all_ranks=True)

        if rank is not None:
            return cls(rank=Rank(rank))
        if is_fbcode():
            from fb.utils import rank_config_from_cli_args

            return rank_config_from_cli_args(cls, source_type)
        elif source_type in [SourceType.LOCAL, SourceType.LOCAL_FILE]:
            return cls(is_local=True)
        else:
            return cls(all_ranks=True)

    def to_rank(self) -> Rank:
        """
        Get the rank object from this config.

        Returns:
            Rank object
        """
        if self.rank:
            return self.rank
        return Rank()


def gzip_folder(folder: str, parsed_ranks: List[str], verbose: bool) -> None:
    """
    Gzip all files in a folder.

    Args:
        folder: Path to folder
        verbose: Whether to print verbose information
    """
    for parsed_rank in parsed_ranks:
        target_folder = os.path.join(folder, parsed_rank)
        for filename in os.listdir(target_folder):
            if filename.endswith(".gz"):
                continue

            file_path = os.path.join(target_folder, filename)
            if os.path.isfile(file_path):
                if verbose:
                    logger.info(f"Gzipping {file_path}")
                with open(file_path, "rb") as f_in:
                    with gzip.open(file_path + ".gz", "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                # Delete the original file after successful compression
                os.remove(file_path)
                if verbose:
                    logger.info(f"Deleted original file {file_path}")
    logger.info(f"Compressed all files in {folder}")


def copy_local_to_tmpdir(local_path: str, verbose: bool = False) -> str:
    """
    Copy local log files to a temporary directory.

    Args:
        local_path: Path to local directory containing logs
        verbose: Whether to print verbose information

    Returns:
        Path to temporary directory containing copied logs
    """
    temp_dir = tempfile.mkdtemp()

    for item in os.listdir(local_path):
        item_path = os.path.join(local_path, item)
        if os.path.isfile(item_path) and os.path.basename(item_path).startswith(
            LOG_PREFIX
        ):
            if verbose:
                logger.info(f"Copying {item_path} to {temp_dir}")
            shutil.copy2(item_path, temp_dir)

    return temp_dir


def parse_logs(
    logs_to_parse: str,
    rank_config: RankConfig,
    verbose: bool = False,
) -> Tuple[str, List[str]]:
    """
    Parse logs.

    Args:
        logs_to_parse: Path to directory containing logs to parse
        rank_config: Rank configuration
        verbose: Whether to print verbose information

    Returns:
        Tuple of (parsed log directory, list of parsed ranks)
    """

    raw_log_dir = logs_to_parse
    parsed_log_dir = tempfile.mkdtemp()

    # Dictionary to store ranks and their log files
    ranks = defaultdict(list)  # Dict[Rank, List[str]]

    # Find all eligible logs in the raw log directory
    for item in os.listdir(raw_log_dir):
        path = os.path.join(raw_log_dir, item)
        if not os.path.isfile(path):
            continue

        log_name = f"{LOG_PREFIX}{rank_config.to_rank().to_string('')}"

        if log_name in item:
            # Check if the log has a rank in its name
            rank_match = LOG_RANK_REGEX.search(item)
            if rank_match:
                # If we have a rank, add it to the list of ranks
                rank_value = int(rank_match.group(1))
                rank = Rank(rank_value)
                ranks[rank].append(path)
            elif rank_config.is_local:
                # Local logs don't always have a rank associated with them, we can push as default
                rank = Rank()
                if rank in ranks:
                    ranks[rank].append(path)
                else:
                    ranks[rank] = [path]

    if not ranks:
        raise RuntimeError(f"No eligible structured trace logs found in {raw_log_dir}")

    parsed_ranks = []

    # Parse each eligible log
    for rank, files in ranks.items():
        use_filenames = False
        if len(files) > 1:
            logger.warning(
                "Warning: multiple logs found for the same rank. Using filenames."
            )
            use_filenames = True

        for file_path in files:
            filename = os.path.basename(file_path)
            input_file = os.path.join(raw_log_dir, filename)

            relative_path = ""
            if use_filenames:
                rank_prefix = "" if rank.is_default else f"{rank.to_string('')}/"
                relative_path = f"{rank_prefix}{filename}"
            else:
                relative_path = rank.to_string("")

            output_dir = os.path.join(parsed_log_dir, relative_path)
            parsed_ranks.append(relative_path)
            parse_single_file(input_file, output_dir)

    return parsed_log_dir, parsed_ranks


def save_logs(out_dir: Path, parsed_logs: str, overwrite: bool, verbose: bool) -> None:
    """
    Save logs to a local directory.

    Args:
        out_dir: Path to output directory
        parsed_logs: Path to directory containing parsed logs
        overwrite: Whether to overwrite existing logs
        verbose: Whether to print verbose information
    """
    if out_dir.exists():
        if not overwrite:
            raise RuntimeError(
                f"{out_dir} already exists, pass --overwrite to overwrite"
            )
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Copying parsed logs from {parsed_logs} to {out_dir}")

    # Copy each item in the parsed_logs directory to the output directory
    for item in os.listdir(parsed_logs):
        src_path = os.path.join(parsed_logs, item)
        dst_path = os.path.join(out_dir, item)

        if os.path.isdir(src_path):
            if verbose:
                logger.info(f"Copying directory {src_path}/ to {dst_path}/")
            shutil.copytree(src_path, dst_path)
        else:
            if verbose:
                logger.info(f"Copying file from {src_path} to {dst_path}")
            shutil.copy2(src_path, dst_path)
