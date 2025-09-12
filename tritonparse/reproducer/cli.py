import argparse


def _add_reproducer_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments for the reproducer to a parser."""
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--line-index",
        type=int,
        help="The line number of the launch event in the input file to reproduce.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save the reproducer script and context JSON."
        "Defaults to 'repro_output/<kernel_name>/' if not provided.",
    )
