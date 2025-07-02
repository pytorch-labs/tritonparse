#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.

from tritonparse.utils import unified_parse_from_cli


# We need this as an entrace for fbpkg
def main():
    unified_parse_from_cli()


if __name__ == "__main__":
    # Do not add code here, it won't be run. Add them to the function called below.
    main()  # pragma: no cover
