import argparse
import sys


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Generate a runnable Triton repro script from a tritonparse NDJSON" " trace"
        )
    )
    p.add_argument("--ndjson", required=True, help="Path to NDJSON trace file")
    p.add_argument(
        "--launch-index",
        type=int,
        default=0,
        help="Launch index to reproduce",
    )
    p.add_argument("--out", default="repro.py", help="Output Python file path")
    p.add_argument(
        "--execute",
        action="store_true",
        help="Execute the generated script",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Auto-repair attempts if execution fails",
    )
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

    # Lazy imports to allow `--help` without optional deps installed
    from .config import load_config
    from .orchestrator import generate_from_ndjson

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
        args.ndjson,
        provider,
        launch_index=args.launch_index,
        out_py=args.out,
        execute=args.execute,
        retries=args.retries,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    print(res)


if __name__ == "__main__":  # pragma: no cover
    main()
