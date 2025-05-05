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
