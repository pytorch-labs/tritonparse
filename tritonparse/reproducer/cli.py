import argparse
import logging
import sys

from .config import load_config
from .factory import make_gemini_provider
from .log_config import setup_logging
from .orchestrator import generate_from_ndjson


def _add_reproducer_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments for the reproducer to a parser."""
    parser.add_argument("--ndjson", required=True)
    # Mode 1: Standard repro from launch index
    parser.add_argument(
        "--launch-index",
        type=int,
        default=0,
        help="Launch index to reproduce (for successful execution mode).",
    )
    # Mode 2: Reproduce a failure
    parser.add_argument(
        "--reproduce-error-from",
        type=str,
        help="Path to the error message file to reproduce.",
    )
    parser.add_argument(
        "--on-line",
        type=int,
        help="The line number of the launch event in the NDJSON file to reproduce.",
    )
    parser.add_argument(
        "--ai-analysis",
        action="store_true",
        help="Enable LLM-based analysis and verification for error reproduction.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=10,
        help="Max number of generation attempts. Defaults to 10.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save the reproducer script and context JSON. "
        "Defaults to 'repro_output/<kernel_name>/'.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate arguments for the reproducer."""
    is_error_repro_mode = args.reproduce_error_from is not None
    if is_error_repro_mode:
        if args.on_line is None:
            parser.error("--on-line is required when using --reproduce-error-from.")
        if args.launch_index != 0:
            # Use logger for warnings
            logging.getLogger(__name__).warning(
                "Warning: --launch-index is ignored when --reproduce-error-from is used."
            )


def add_reproducer_subparser(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="subcommand")
    repro = sub.add_parser(
        "repro",
        help="Generate a runnable Triton repro script from NDJSON",
    )
    _add_reproducer_args(repro)


def maybe_handle_reproducer(args: argparse.Namespace) -> bool:
    if getattr(args, "subcommand", None) != "repro":
        return False

    # Setup logging as early as possible
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    # Argument validation is now handled by the parser in __main__ for direct execution,
    # but needs to be called explicitly when used as a subcommand.
    # We need a dummy parser to call .error() if validation fails.
    # A better approach might be to raise an exception. For now, this is simpler.
    _validate_args(argparse.ArgumentParser(), args)

    cfg = load_config()
    provider = None
    is_error_repro_mode = args.reproduce_error_from is not None

    # Lazily initialize the provider only if AI is needed.
    # AI is needed in success mode (for the repair loop) or
    # if explicitly enabled in error repro mode.
    if not is_error_repro_mode or args.ai_analysis:
        provider = make_gemini_provider()

    # Pass all relevant args to the orchestrator
    res = generate_from_ndjson(
        ndjson_path=args.ndjson,
        provider=provider,
        out_dir=args.out_dir,
        execute=args.execute,
        # Mode-specific params
        launch_index=args.launch_index,
        reproduce_error_from=args.reproduce_error_from,
        on_line=args.on_line,
        attempts=args.attempts,
        ai_analysis=args.ai_analysis,
        # LLM params
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    # The final result is still printed to stdout as it's the program's output
    print(res)
    return True
