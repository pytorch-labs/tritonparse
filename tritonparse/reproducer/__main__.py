import argparse

from tritonparse.reproducer import cli
from tritonparse.reproducer.orchestrator import reproducer


def main():
    parser = argparse.ArgumentParser()
    cli._add_reproducer_args(parser)
    args = parser.parse_args()
    reproducer(args.ndjson, args.line_index, args.out_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
