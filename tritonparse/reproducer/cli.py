import argparse

from .config import load_config
from .factory import make_gemini_provider
from .orchestrator import generate_from_ndjson


def add_reproducer_subparser(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="subcommand")
    repro = sub.add_parser(
        "repro",
        help="Generate a runnable Triton repro script from NDJSON",
    )
    repro.add_argument("--ndjson", required=True)
    repro.add_argument("--launch-index", type=int, default=0)
    repro.add_argument("--out", default="repro.py")
    repro.add_argument("--execute", action="store_true")
    repro.add_argument("--retries", type=int, default=0)


def maybe_handle_reproducer(args: argparse.Namespace) -> bool:
    if getattr(args, "subcommand", None) != "repro":
        return False
    cfg = load_config()
    provider = make_gemini_provider()
    res = generate_from_ndjson(
        args.ndjson,
        provider,
        launch_index=args.launch_index,
        out_py=args.out,
        execute=args.execute,
        retries=args.retries,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    print(res)
    return True
