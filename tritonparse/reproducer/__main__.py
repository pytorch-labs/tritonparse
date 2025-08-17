import argparse
import logging
import sys

# Lazy imports are moved to the bottom
from .cli import _add_reproducer_args, _validate_args
from .config import load_config
from .log_config import setup_logging
from .orchestrator import generate_from_ndjson


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Generate a runnable Triton repro script from a tritonparse NDJSON" " trace"
        )
    )
    # Add all arguments from the centralized helper function
    _add_reproducer_args(p)

    # Add arguments specific to direct execution via __main__
    p.add_argument(
        "--temperature",
        type=float,
        help="Override sampling temperature",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        help="Override max tokens for generation",
    )
    args = p.parse_args()

    # Setup logging as early as possible
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    # Validate arguments using the centralized helper function
    _validate_args(p, args)

    try:
        from .factory import make_gemini_provider
    except Exception:  # pragma: no cover
        print(
            "Failed to import provider factory. Ensure optional deps are installed (e.g. google-genai).",
            file=sys.stderr,
        )
        raise

    cfg = load_config()
    try:
        provider = make_gemini_provider()
    except ModuleNotFoundError:  # pragma: no cover
        print(
            "Gemini provider requires 'google-genai'. Install via: pip install google-genai",
            file=sys.stderr,
        )
        sys.exit(2)
    temperature = args.temperature if args.temperature is not None else cfg.temperature
    max_tokens = args.max_tokens if args.max_tokens is not None else cfg.max_tokens

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
        # LLM params
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # The final result is still printed to stdout as it's the program's output
    print(res)


if __name__ == "__main__":  # pragma: no cover
    main()
